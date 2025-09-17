from multiprocessing import reduction
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from dataset import save_emb
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
    
class RMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        # 均方根归一化
        norm = x.norm(2, dim=-1, keepdim=True) * (1.0 / (x.size(-1) ** 0.5))
        return self.weight * (x / (norm + self.eps))

class FlashMultiHeadAttention(torch.nn.Module):
    def __init__(self, hidden_units, num_heads, dropout_rate=0.0):
        super(FlashMultiHeadAttention, self).__init__()

        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.head_dim = hidden_units // num_heads
        self.dropout_rate = dropout_rate

        assert hidden_units % num_heads == 0, "hidden_units must be divisible by num_heads"

        self.q_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.k_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.v_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.out_linear = torch.nn.Linear(hidden_units, hidden_units)

    def forward(self, query, key, value, attn_mask=None):
        batch_size, seq_len, _ = query.size()

        # 计算Q, K, V
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)

        # reshape为multi-head格式
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if hasattr(F, 'scaled_dot_product_attention'):
            # PyTorch 2.0+ 使用内置的Flash Attention
            attn_output = F.scaled_dot_product_attention(
                Q, K, V, dropout_p=self.dropout_rate if self.training else 0.0, attn_mask=attn_mask.unsqueeze(1)
            )
            attn_output = torch.nan_to_num(attn_output, nan=0.0)
        else:
            # 降级到标准注意力机制
            scale = (self.head_dim) ** -0.5
            scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

            if attn_mask is not None:
                scores.masked_fill_(attn_mask.unsqueeze(1).logical_not(), float('-1e9'))

            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = F.dropout(attn_weights, p=self.dropout_rate, training=self.training)
            attn_output = torch.matmul(attn_weights, V)

        # reshape回原来的格式
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_units)

        # 最终的线性变换
        output = self.out_linear(attn_output)

        return output, None

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.GELU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, Length)
        return outputs

class HSTU(nn.Module):
    """
    HSTU 模块 (phi1, phi2 = SiLU, 不含 rab 和 dropout)
    """
    def __init__(self, in_dim, hidden_dim, out_dim=None, num_heads=1, eps=1e-8):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim or in_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.eps = eps

        # 四个独立线性层
        self.U_proj = nn.Linear(in_dim, hidden_dim)
        self.V_proj = nn.Linear(in_dim, hidden_dim)
        self.Q_proj = nn.Linear(in_dim, hidden_dim)
        self.K_proj = nn.Linear(in_dim, hidden_dim)

        self.phi1_act = nn.SiLU()   # phi1 = SiLU
        self.phi2_act = nn.SiLU()   # phi2 = SiLU

        # f2: final projection
        self.f2 = nn.Linear(hidden_dim, self.out_dim)

        # LayerNorm applied on AV before element-wise multiply with U
        self.norm = nn.RMSNorm(hidden_dim)

    def _reshape_for_heads(self, x):
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def _reshape_from_heads(self, x):
        B, H, T, D = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(B, T, H * D)

    def forward(self, x, attn_mask=None):
        """
        x: (B, T, C_in)
        attn_mask: None or tensor boolean/float mask broadcastable to (B, H, T, T) or (B, T, T).
                   mask values: 1/True -> keep, 0/False -> mask out.
        returns y: (B, T, C_out)
        """
        B, T, C = x.shape

        # 分别线性映射 + phi1
        U = self.phi1_act(self.U_proj(x))  # (B, T, hidden_dim)
        V = self.phi1_act(self.V_proj(x))
        Q = self.phi1_act(self.Q_proj(x))
        K = self.phi1_act(self.K_proj(x))

        # multi-head reshape
        Qh = self._reshape_for_heads(Q)
        Kh = self._reshape_for_heads(K)
        Vh = self._reshape_for_heads(V)

        # attention scores
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_logits = torch.matmul(Qh, Kh.transpose(-2, -1)) * scale
        # phi2
        A = self.phi2_act(attn_logits).clamp_min(0.)
        if attn_mask is not None:
            A = A.masked_fill(attn_mask.unsqueeze(1).logical_not(), 0.0)
        # normalize A
        denom = A.sum(dim=-1, keepdim=True)
        A = torch.where(denom > 1e-12, A / (denom + self.eps), torch.zeros_like(A))

        # A @ V
        AVh = torch.matmul(A, Vh)
        AV = self._reshape_from_heads(AVh)

        # Norm + element-wise multiply
        AV_norm = self.norm(AV)
        UV = AV_norm * U

        # f2 projection
        y = self.f2(UV)
        return y

class BaselineModel(torch.nn.Module):
    """
    Args:
        user_num: 用户数量
        item_num: 物品数量
        feat_statistics: 特征统计信息，key为特征ID，value为特征数量
        feat_types: 各个特征的特征类型，key为特征类型名称，value为包含的特征ID列表，包括user和item的sparse, array, emb, continual类型
        args: 全局参数

    Attributes:
        user_num: 用户数量
        item_num: 物品数量
        dev: 设备
        norm_first: 是否先归一化
        maxlen: 序列最大长度
        item_emb: Item Embedding Table
        user_emb: User Embedding Table
        sparse_emb: 稀疏特征Embedding Table
        emb_transform: 多模态特征的线性变换
        userdnn: 用户特征拼接后经过的全连接层
        itemdnn: 物品特征拼接后经过的全连接层
    """

    def __init__(self, user_num, item_num, feat_statistics, feat_types, args):  #
        super(BaselineModel, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.norm_first = args.norm_first
        self.maxlen = args.maxlen
        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch

        self.item_emb = torch.nn.Embedding(self.item_num + 1, args.embedding_dim, padding_idx=0)
        self.user_emb = torch.nn.Embedding(self.user_num + 1, args.embedding_dim, padding_idx=0)

        self.pos_emb = torch.nn.Embedding(2 * args.maxlen + 1, args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.emb_layernorm = torch.nn.RMSNorm(args.hidden_units, eps=1e-8)
        self.sparse_emb = torch.nn.ModuleDict()
        self.emb_transform = torch.nn.ModuleDict()

        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self._init_feat_info(feat_statistics, feat_types)

        userdim = args.embedding_dim * (len(self.USER_SPARSE_FEAT) + 1 + len(self.USER_ARRAY_FEAT)) + len(
            self.USER_CONTINUAL_FEAT
        )
        self.item_filed_num_static = len(self.ITEM_SPARSE_FEAT) + 1 + len(self.ITEM_ARRAY_FEAT) + len(self.ITEM_EMB_FEAT)
        item_dim_static = args.embedding_dim * self.item_filed_num_static

        self.item_filed_num_dynamic = self.item_filed_num_static + 7
        item_dim_dynamic = args.embedding_dim * self.item_filed_num_dynamic + len(self.ITEM_CONTINUAL_FEAT)

        self.userdnn = torch.nn.Linear(userdim, args.hidden_units)
        # self.itemdnn = torch.nn.Linear(itemdim, args.hidden_units)

        self.item_static_dnn = torch.nn.Linear(item_dim_static, args.hidden_units)
        self.item_dynamic_dnn = torch.nn.Linear(item_dim_dynamic, args.hidden_units)

        self.dnn_user_layernorm = torch.nn.RMSNorm(args.hidden_units, eps=1e-8)
        self.dnn_item_layernorm = torch.nn.RMSNorm(args.hidden_units, eps=1e-8)

        self.last_layernorm = torch.nn.RMSNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.RMSNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = HSTU(
                args.hidden_units, args.hidden_units, args.hidden_units, args.num_heads
            )  # 优化：用FlashAttention替代标准Attention
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.RMSNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

        for k in self.USER_SPARSE_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.USER_SPARSE_FEAT[k] + 1, args.embedding_dim, padding_idx=0)
        for k in self.ITEM_SPARSE_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.ITEM_SPARSE_FEAT[k] + 1, args.embedding_dim, padding_idx=0)
        for k in self.ITEM_ARRAY_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.ITEM_ARRAY_FEAT[k] + 1, args.embedding_dim, padding_idx=0)
        for k in self.USER_ARRAY_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.USER_ARRAY_FEAT[k] + 1, args.embedding_dim, padding_idx=0)
        for k in self.ITEM_EMB_FEAT:
            self.emb_transform[k] = torch.nn.Linear(self.ITEM_EMB_FEAT[k], args.embedding_dim)
        
        # 时间特征处理模块
        # 为时间特征创建Embedding层
        self.time_emb = torch.nn.ModuleDict()
        self.time_emb['weekday'] = torch.nn.Embedding(8, args.embedding_dim, padding_idx=0)  # 0-6
        self.time_emb['hour'] = torch.nn.Embedding(25, args.embedding_dim, padding_idx=0)  # 0-23
        self.time_emb['time_gap'] = torch.nn.Embedding(16, args.embedding_dim, padding_idx=0)  # 0-20
        self.time_emb['action_type'] = torch.nn.Embedding(3, args.embedding_dim, padding_idx=0)  # 0-2
        self.time_emb['is_weekend'] = torch.nn.Embedding(3, args.embedding_dim, padding_idx=0)  # 0-2
        self.time_emb['day_of_year'] = torch.nn.Embedding(366, args.embedding_dim, padding_idx=0)  # 0-365
        self.time_emb['week_of_year'] = torch.nn.Embedding(53, args.embedding_dim, padding_idx=0)  # 0-52

    def _init_feat_info(self, feat_statistics, feat_types):
        """
        将特征统计信息（特征数量）按特征类型分组产生不同的字典，方便声明稀疏特征的Embedding Table

        Args:
            feat_statistics: 特征统计信息，key为特征ID，value为特征数量
            feat_types: 各个特征的特征类型，key为特征类型名称，value为包含的特征ID列表，包括user和item的sparse, array, emb, continual类型
        """
        self.USER_SPARSE_FEAT = {k: feat_statistics[k] for k in feat_types['user_sparse']}
        self.USER_CONTINUAL_FEAT = feat_types['user_continual']
        self.ITEM_SPARSE_FEAT = {k: feat_statistics[k] for k in feat_types['item_sparse']}
        self.ITEM_CONTINUAL_FEAT = feat_types['item_continual']
        self.USER_ARRAY_FEAT = {k: feat_statistics[k] for k in feat_types['user_array']}
        self.ITEM_ARRAY_FEAT = {k: feat_statistics[k] for k in feat_types['item_array']}
        EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
        self.ITEM_EMB_FEAT = {k: EMB_SHAPE_DICT[k] for k in feat_types['item_emb']}  # 记录的是不同多模态特征的维度

    def feat2tensor(self, seq_feature, k):
        """
        Args:
            seq_feature: 序列特征list，每个元素为当前时刻的特征字典，形状为 [batch_size, maxlen]
            k: 特征ID

        Returns:
            batch_data: 特征值的tensor，形状为 [batch_size, maxlen, max_array_len(if array)]
        """
        batch_size = len(seq_feature)
        if k in self.ITEM_ARRAY_FEAT or k in self.USER_ARRAY_FEAT:
            # 如果特征是Array类型，需要先对array进行padding，然后转换为tensor
            max_array_len = 0
            max_seq_len = 0

            for i in range(batch_size):
                seq_data = [item[k] for item in seq_feature[i]]
                max_seq_len = max(max_seq_len, len(seq_data))
                max_array_len = max(max_array_len, max(len(item_data) for item_data in seq_data))

            batch_data = np.zeros((batch_size, max_seq_len, max_array_len), dtype=np.int64)
            for i in range(batch_size):
                seq_data = [item[k] for item in seq_feature[i]]
                for j, item_data in enumerate(seq_data):
                    actual_len = min(len(item_data), max_array_len)
                    batch_data[i, j, :actual_len] = item_data[:actual_len]

            return torch.from_numpy(batch_data).to(self.dev)
        else:
            # 如果特征是Sparse类型，直接转换为tensor
            max_seq_len = max(len(seq_feature[i]) for i in range(batch_size))
            batch_data = np.zeros((batch_size, max_seq_len), dtype=np.int64)

            for i in range(batch_size):
                seq_data = [item[k] for item in seq_feature[i]]
                batch_data[i] = seq_data

            return torch.from_numpy(batch_data).to(self.dev)

    def feat2emb(
        self, seq, feature_array, mask=None, 
        include_user=False, include_time=False, time_feat=None, ctr_feat=None
    ):
        """
        Args:
            seq: 序列ID
            feature_array: 特征list，每个元素为当前时刻的特征字典
            mask: 掩码，1表示item，2表示user
            include_user: 是否处理用户特征
            include_time: 是否处理时间特征
            time_feat: 时间特征字典，包含weekday, hour, time_gap

        Returns:
            seqs_emb: 序列特征的Embedding
        """
        seq = seq.to(self.dev)

        # 处理 user/item embedding
        if include_user:
            user_mask = (mask == 2).to(self.dev)
            item_mask = (mask == 1).to(self.dev)
            user_embedding = self.user_emb(user_mask * seq)
            item_embedding = self.item_emb(item_mask * seq)
            item_feat_list = [item_embedding]
            user_feat_list = [user_embedding]
        else:
            item_embedding = self.item_emb(seq)
            item_feat_list = [item_embedding]

        # 处理各种特征
        # ===== 处理各种特征 =====
        all_feat_types = [
            (self.ITEM_SPARSE_FEAT, 'item_sparse', item_feat_list),
            (self.ITEM_ARRAY_FEAT, 'item_array', item_feat_list),
        ]
        if include_time:  # 只有在 include_time 时才加入 item_continual
            all_feat_types.append(
                (self.ITEM_CONTINUAL_FEAT, 'item_continual', item_feat_list)
            )

        if include_user:
            all_feat_types.extend(
                [
                    (self.USER_SPARSE_FEAT, 'user_sparse', user_feat_list),
                    (self.USER_ARRAY_FEAT, 'user_array', user_feat_list),
                    (self.USER_CONTINUAL_FEAT, 'user_continual', user_feat_list),
                ]
            )

        for feat_dict, feat_type, feat_list in all_feat_types:
            if not feat_dict:
                continue
            for k in feat_dict:
                tensor_feature = feature_array[k].to(self.dev)
                if feat_type.endswith('sparse'):
                    feat_list.append(self.sparse_emb[k](tensor_feature))
                elif feat_type.endswith('array'):
                    feat_list.append(self.sparse_emb[k](tensor_feature).sum(2))
                elif feat_type.endswith('continual'):
                    feat_list.append(tensor_feature.unsqueeze(2))

        for k in self.ITEM_EMB_FEAT:
            tensor_feature = feature_array[k].to(self.dev)
            item_feat_list.append(self.emb_transform[k](tensor_feature))

        # ===== 新逻辑：处理时间特征 =====
        if include_time and time_feat is not None:
            time_emb_list = []
            for time_key in ['weekday', 'is_weekend', 'hour', 'day_of_year', 'week_of_year', 'time_gap', 'action_type']:
                if time_key in time_feat:
                    time_tensor = time_feat[time_key].to(self.dev)
                    time_emb = self.time_emb[time_key](time_tensor)
                    # print(time_emb.shape)
                    time_emb_list.append(time_emb)

            if time_emb_list:
                all_time_emb = torch.cat(time_emb_list, dim=2)
                item_feat_list.append(all_time_emb)

            all_item_emb = torch.cat(item_feat_list, dim=2)
            all_item_emb = F.gelu(self.item_dynamic_dnn(all_item_emb))

        else:  # 走 static dnn
            all_item_emb = torch.cat(item_feat_list, dim=2)
            all_item_emb = F.gelu(self.item_static_dnn(all_item_emb))

        # ===== 用户特征处理 =====
        if include_user:
            all_user_emb = torch.cat(user_feat_list, dim=2)
            all_user_emb = F.gelu(self.userdnn(all_user_emb))
            seqs_emb = all_item_emb + all_user_emb
        else:
            seqs_emb = all_item_emb

        return seqs_emb

    def log2feats(self, log_seqs, mask, seq_feature, time_feat=None):
        """
        Args:
            log_seqs: 序列ID
            mask: token类型掩码，1表示item token，2表示user token
            seq_feature: 序列特征list，每个元素为当前时刻的特征字典
            time_feat: 时间特征字典

        Returns:
            seqs_emb: 序列的Embedding，形状为 [batch_size, maxlen, hidden_units]
        """
        batch_size = log_seqs.shape[0]
        maxlen = log_seqs.shape[1]
        seqs = self.feat2emb(log_seqs, seq_feature, mask=mask, include_user=True, include_time=True, time_feat=time_feat, ctr_feat=time_feat.get("user_ctr", None))
        seqs *= self.item_emb.embedding_dim**0.5
        poss = torch.arange(1, maxlen + 1, device=self.dev).unsqueeze(0).expand(batch_size, -1).clone()
        poss *= log_seqs != 0
        seqs += self.pos_emb(poss)
        seqs = self.emb_dropout(seqs)
        seqs = self.emb_layernorm(seqs)

        maxlen = seqs.shape[1]
        ones_matrix = torch.ones((maxlen, maxlen), dtype=torch.bool, device=self.dev)
        attention_mask_tril = torch.tril(ones_matrix)
        attention_mask_pad = (mask != 0).to(self.dev)
        attention_mask = attention_mask_tril.unsqueeze(0) & attention_mask_pad.unsqueeze(1)

        for i in range(len(self.attention_layers)):
            if self.norm_first:
                x = self.attention_layernorms[i](seqs)
                mha_outputs = self.attention_layers[i](x, attention_mask)
                seqs = seqs + mha_outputs
                seqs = seqs + self.forward_layers[i](self.forward_layernorms[i](seqs))
            else:
                mha_outputs = self.attention_layers[i](seqs, attention_mask)
                seqs = self.attention_layernorms[i](seqs + mha_outputs)
                seqs = self.forward_layernorms[i](seqs + self.forward_layers[i](seqs))

        log_feats = self.last_layernorm(seqs)

        return log_feats
    
    def forward(
        self, user_item, pos_seqs, neg_seqs, mask, next_mask, next_action_type, seq_feature, pos_feature, neg_feature, time_feat
    ):
        """
        训练时调用，计算正负样本的logits

        Args:
            user_item: 用户序列
            pos_seqs: 正样本序列ID
            neg_seqs: 负样本序列ID
            mask: token类型掩码，1表示item token，2表示user token
            next_mask: 下一个token类型掩码，1表示item token，2表示user token
            next_action_type: 下一个token动作类型，0表示曝光，1表示点击
            seq_feature: 序列特征list，每个元素为当前时刻的特征字典
            pos_feature: 正样本特征list，每个元素为当前时刻的特征字典
            neg_feature: 负样本特征list，每个元素为当前时刻的特征字典

        Returns:
            pos_logits: 正样本logits，形状为 [batch_size, maxlen]
            neg_logits: 负样本logits，形状为 [batch_size, maxlen]
        """
        log_feats = self.log2feats(user_item, mask, seq_feature, time_feat)
        loss_mask = (next_mask == 1).to(self.dev)

        pos_embs = self.feat2emb(pos_seqs, pos_feature, include_user=False, include_time=False, time_feat=time_feat)
        neg_embs = self.feat2emb(neg_seqs, neg_feature, include_user=False, include_time=False, time_feat=time_feat)

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)
        pos_logits = pos_logits * loss_mask
        neg_logits = neg_logits * loss_mask

        anchor_emb = log_feats[:, -1, :]
        pos_emb = pos_embs[:, -1, :]
        neg_emb = neg_embs[:, -1, :]

        return pos_logits, neg_logits, anchor_emb, pos_emb, neg_emb

    def predict(self, log_seqs, seq_feature, mask, time_feat=None):
        """
        计算用户序列的表征
        Args:
            log_seqs: 用户序列ID
            seq_feature: 序列特征list，每个元素为当前时刻的特征字典
            mask: token类型掩码，1表示item token，2表示user token
        Returns:
            final_feat: 用户序列的表征，形状为 [batch_size, hidden_units]
        """
        log_feats = self.log2feats(log_seqs, mask, seq_feature, time_feat)

        final_feat = log_feats[:, -1, :]
        final_feat = final_feat / (final_feat.norm(dim=-1, keepdim=True) + 1e-8)  # 归一化

        return final_feat
    
    def _count_fields(self):
        """
        统计 item 和 user 各自的 field 数量
        """
        # item fields
        num_item_fields = 1  # item_id 本身
        num_item_fields += len(self.ITEM_SPARSE_FEAT)
        num_item_fields += len(self.ITEM_ARRAY_FEAT)
        num_item_fields += len(self.ITEM_CONTINUAL_FEAT)
        num_item_fields += len(self.ITEM_EMB_FEAT)

        # user fields
        num_user_fields = 1  # user_id 本身
        num_user_fields += len(self.USER_SPARSE_FEAT)
        num_user_fields += len(self.USER_ARRAY_FEAT)
        num_user_fields += len(self.USER_CONTINUAL_FEAT)

        return num_item_fields, num_user_fields

    def feat2emb4save(self, seq, feature_array, mask=None, include_user=False, include_time=False, time_feat=None, ctr_feat=None):
        """
        Args:
            seq: 序列ID
            feature_array: 特征list，每个元素为当前时刻的特征字典
            mask: 掩码，1表示item，2表示user
            include_user: 是否处理用户特征，在两种情况下不打开：1) 训练时在转换正负样本的特征时（因为正负样本都是item）;2) 生成候选库item embedding时。

        Returns:
            seqs_emb: 序列特征的Embedding
        """
        seq = seq.to(self.dev)
        # pre-compute embedding
        if include_user:
            user_mask = (mask == 2).to(self.dev)
            item_mask = (mask == 1).to(self.dev)
            user_embedding = self.user_emb(user_mask * seq)
            item_embedding = self.item_emb(item_mask * seq)
            item_feat_list = [item_embedding]
            user_feat_list = [user_embedding]
        else:
            item_embedding = self.item_emb(seq)
            item_feat_list = [item_embedding]

        # batch-process all feature types
        all_feat_types = [
            (self.ITEM_SPARSE_FEAT, 'item_sparse', item_feat_list),
            (self.ITEM_ARRAY_FEAT, 'item_array', item_feat_list),
        ]
        if include_time:  # 只有在 include_time 时才加入 item_continual
            all_feat_types.append(
                (self.ITEM_CONTINUAL_FEAT, 'item_continual', item_feat_list)
            )

        if include_user:
            all_feat_types.extend(
                [
                    (self.USER_SPARSE_FEAT, 'user_sparse', user_feat_list),
                    (self.USER_ARRAY_FEAT, 'user_array', user_feat_list),
                    (self.USER_CONTINUAL_FEAT, 'user_continual', user_feat_list),
                ]
            )

        # batch-process each feature type
        for feat_dict, feat_type, feat_list in all_feat_types:
            if not feat_dict:
                continue

            for k in feat_dict:
                tensor_feature = self.feat2tensor(feature_array, k)

                if feat_type.endswith('sparse'):
                    feat_list.append(self.sparse_emb[k](tensor_feature))
                elif feat_type.endswith('array'):
                    feat_list.append(self.sparse_emb[k](tensor_feature).sum(2))
                elif feat_type.endswith('continual'):
                    feat_list.append(tensor_feature.unsqueeze(2))

        for k in self.ITEM_EMB_FEAT:
            # collect all data to numpy, then batch-convert
            batch_size = len(feature_array)
            emb_dim = self.ITEM_EMB_FEAT[k]
            seq_len = len(feature_array[0])

            # pre-allocate tensor
            batch_emb_data = np.zeros((batch_size, seq_len, emb_dim), dtype=np.float32)

            for i, seq in enumerate(feature_array):
                for j, item in enumerate(seq):
                    if k in item:
                        batch_emb_data[i, j] = item[k]

            # batch-convert and transfer to GPU
            tensor_feature = torch.from_numpy(batch_emb_data).to(self.dev)
            item_feat_list.append(self.emb_transform[k](tensor_feature))
        # ===== 新逻辑：处理时间特征 =====
        if include_time and time_feat is not None:
            time_emb_list = []
            for time_key in ['weekday', 'is_weekend', 'hour', 'day_of_year', 'week_of_year', 'time_gap', 'action_type']:
                if time_key in time_feat:
                    time_tensor = time_feat[time_key].to(self.dev)
                    time_emb = self.time_emb[time_key](time_tensor)
                    time_emb_list.append(time_emb)

            if time_emb_list:
                all_time_emb = torch.cat(time_emb_list, dim=2)
                item_feat_list.append(all_time_emb)

            all_item_emb = torch.cat(item_feat_list, dim=2)
            all_item_emb = F.gelu(self.item_dynamic_dnn(all_item_emb))

        else:  # 走 static dnn
            all_item_emb = torch.cat(item_feat_list, dim=2)
            all_item_emb = F.gelu(self.item_static_dnn(all_item_emb))

        # ===== 用户特征处理 =====
        if include_user:
            all_user_emb = torch.cat(user_feat_list, dim=2)
            all_user_emb = F.gelu(self.userdnn(all_user_emb))
            seqs_emb = all_item_emb + all_user_emb
        else:
            seqs_emb = all_item_emb

        return seqs_emb

    def save_item_emb(self, item_ids, retrieval_ids, feat_dict, save_path, time_feat=None, batch_size=1024):
        """
        生成候选库item embedding，用于检索

        Args:
            item_ids: 候选item ID（re-id形式）
            retrieval_ids: 候选item ID（检索ID，从0开始编号，检索脚本使用）
            feat_dict: 训练集所有item特征字典，key为特征ID，value为特征值
            save_path: 保存路径
            batch_size: 批次大小
        """
        all_embs = []

        for start_idx in tqdm(range(0, len(item_ids), batch_size), desc="Saving item embeddings"):
            end_idx = min(start_idx + batch_size, len(item_ids))

            item_seq = torch.tensor(item_ids[start_idx:end_idx], device=self.dev).unsqueeze(0)
            batch_feat = []
            for i in range(start_idx, end_idx):
                batch_feat.append(feat_dict[i])

            batch_feat = np.array(batch_feat, dtype=object)

            batch_emb = self.feat2emb4save(item_seq, [batch_feat], include_user=False, include_time=False, time_feat=time_feat).squeeze(0)
            batch_emb = batch_emb / (batch_emb.norm(dim=-1, keepdim=True) + 1e-8)  # 归一化
            all_embs.append(batch_emb.detach().cpu().numpy().astype(np.float32))

        # 合并所有批次的结果并保存
        final_ids = np.array(retrieval_ids, dtype=np.uint64).reshape(-1, 1)
        final_embs = np.concatenate(all_embs, axis=0)
        save_emb(final_embs, Path(save_path, 'embedding.fbin'))
        save_emb(final_ids, Path(save_path, 'id.u64bin'))
    
        return final_embs
    def compute_infonce_loss(
        self, user_item, pos_seqs, neg_seqs, mask, next_mask, next_action_type,
        seq_feature, pos_feature, neg_feature, writer, time_feat=None,
        temperature=0.05, chunk_size=1024
        ):
        seq_emb = self.log2feats(user_item, mask, seq_feature, time_feat)
        loss_mask = (next_mask == 1).to(self.dev)

        pos_emb = self.feat2emb(pos_seqs, pos_feature, include_user=False)
        neg_emb = self.feat2emb(neg_seqs, neg_feature, include_user=False)

        hidden_dim = neg_emb.size(-1)

        eps = 1e-8
        seq_emb = seq_emb / (seq_emb.norm(dim=-1, keepdim=True) + eps)
        pos_emb = pos_emb / (pos_emb.norm(dim=-1, keepdim=True) + eps)
        neg_emb = neg_emb / (neg_emb.norm(dim=-1, keepdim=True) + eps)

        # 正样本得分
        pos_logits = F.cosine_similarity(seq_emb, pos_emb, dim=-1).unsqueeze(-1)
        pos_logits = pos_logits - 0.08

        # 负样本 reshape
        neg_emb_all = neg_emb.reshape(-1, hidden_dim)

        # 分块计算 neg_logits
        neg_logits_chunks = []
        for i in range(0, neg_emb_all.size(0), chunk_size):
            chunk = neg_emb_all[i:i+chunk_size]  # [chunk, hidden_dim]
            logits_chunk = torch.matmul(seq_emb, chunk.transpose(0, 1))  # [B, chunk]
            neg_logits_chunks.append(logits_chunk)
        neg_logits = torch.cat(neg_logits_chunks, dim=-1)

        writer.add_scalar('Model/neg_logits_mean', neg_logits.mean().item())
        writer.add_scalar('Model/pos_logits_mean', pos_logits.mean().item())

        logits = torch.cat([pos_logits, neg_logits], dim=-1)
        logits = logits[loss_mask.bool()] / temperature
        labels = torch.zeros(logits.size(0), dtype=torch.int64, device=logits.device)

        loss = F.cross_entropy(logits, labels)
        return loss