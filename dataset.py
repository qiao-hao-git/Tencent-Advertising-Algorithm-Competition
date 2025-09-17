from audioop import reverse
import json
import pickle
from sqlite3.dbapi2 import Timestamp
import struct
from pathlib import Path
import datetime
import numpy as np
import torch
from tqdm import tqdm
import sys

    
class MyDataset(torch.utils.data.Dataset):
    """
    用户序列数据集
    """

    def __init__(self, data_dir, args):
        super().__init__()
        self.data_dir = Path(data_dir)
        self._load_offsets()   # 只加载偏移量
        self.maxlen = args.maxlen
        self.mm_emb_ids = args.mm_emb_id

        self.item_feat_dict = json.load(open(Path(data_dir, "item_feat_dict.json"), 'r'))
        self.mm_emb_dict = load_mm_emb(Path(data_dir, "creative_emb"), self.mm_emb_ids)
        with open(self.data_dir / 'indexer.pkl', 'rb') as ff:
            indexer = pickle.load(ff)
            self.itemnum = len(indexer['i'])
            self.usernum = len(indexer['u'])
        self.indexer_i_rev = {v: k for k, v in indexer['i'].items()}
        self.indexer_u_rev = {v: k for k, v in indexer['u'].items()}
        self.indexer = indexer

        self.feature_default_value, self.feature_types, self.feat_statistics = self._init_feat_info()

        # 文件路径，而不是直接打开
        self.data_file_path = self.data_dir / "seq.jsonl"

    def _load_offsets(self):
        """只加载偏移量"""
        with open(Path(self.data_dir, 'seq_offsets.pkl'), 'rb') as f:
            self.seq_offsets = pickle.load(f)

    def _load_user_data(self, uid):
        """
        从数据文件中加载单个用户的数据
        """
        with open(self.data_file_path, 'rb') as f:  # 每次临时打开，避免 pickle 错误
            f.seek(self.seq_offsets[uid])
            line = f.readline()
        data = json.loads(line)
        return data

    def _random_neq(self, l, r, s):
        """
        生成一个不在序列s中的随机整数, 用于训练时的负采样

        Args:
            l: 随机整数的最小值
            r: 随机整数的最大值
            s: 序列

        Returns:
            t: 不在序列s中的随机整数
        """
        t = np.random.randint(l, r)
        while t in s or str(t) not in self.item_feat_dict:
            t = np.random.randint(l, r)
        return t

    def __getitem__(self, uid):
        """
        获取单个用户的数据，并进行padding处理，生成模型需要的数据格式

        Args:
            uid: 用户ID(reid)

        Returns:
            seq: 用户序列ID
            pos: 正样本ID（即下一个真实访问的item）
            neg: 负样本ID
            token_type: 用户序列类型，1表示item，2表示user
            next_token_type: 下一个token类型，1表示item，2表示user
            seq_feat: 用户序列特征，每个元素为字典，key为特征ID，value为特征值
            pos_feat: 正样本特征，每个元素为字典，key为特征ID，value为特征值
            neg_feat: 负样本特征，每个元素为字典，key为特征ID，value为特征值
        """
        user_sequence = self._load_user_data(uid)  # 动态加载用户数据

        ext_user_sequence = []
        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, action_type, ts = record_tuple

            if u and user_feat:
                ext_user_sequence.insert(0, (u, user_feat, 2, 0, 0))
            if i and item_feat:
                ext_user_sequence.append((i, item_feat, 1, action_type, ts or 0))
    
        seq = np.zeros([self.maxlen + 1], dtype=np.int32)
        pos = np.zeros([self.maxlen + 1], dtype=np.int32)
        neg = np.zeros([self.maxlen + 1], dtype=np.int32)
        action_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        next_token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        next_action_type = np.zeros([self.maxlen + 1], dtype=np.int32)

        seq_feat = np.empty([self.maxlen + 1], dtype=object)
        pos_feat = np.empty([self.maxlen + 1], dtype=object)
        neg_feat = np.empty([self.maxlen + 1], dtype=object)
        
        # 初始化time_feat数组
        timestamp = np.empty([self.maxlen + 1], dtype=np.int64)
        nxt = ext_user_sequence[-1]
        idx = self.maxlen

        ts = set()
        for record_tuple in ext_user_sequence:
            if record_tuple[2] == 1 and record_tuple[0]:
                ts.add(record_tuple[0])

        for record_tuple in reversed(ext_user_sequence[:-1]):
            i, feat, type_, act_type, ts_now = record_tuple
            next_i, next_feat, next_type, next_act_type, _ = nxt
            feat = self.fill_missing_feat(feat, i)
            next_feat = self.fill_missing_feat(next_feat, next_i)

            seq[idx] = i
            token_type[idx] = type_
            next_token_type[idx] = next_type
            
            if next_act_type is not None:
                next_action_type[idx] = next_act_type
            seq_feat[idx] = feat
            timestamp[idx] = ts_now
            if act_type is not None:
                action_type[idx] = act_type
            
            if next_type == 1 and next_i != 0:
                pos[idx] = next_i
                pos_feat[idx] = next_feat
                neg_id = self._random_neq(1, self.itemnum + 1, ts)
                neg[idx] = neg_id
                neg_feat[idx] = self.fill_missing_feat(self.item_feat_dict[str(neg_id)], neg_id)
            nxt = record_tuple
            idx -= 1
            if idx == -1:
                break
        
        seq_feat = np.where(seq_feat == None, self.feature_default_value, seq_feat)
        pos_feat = np.where(pos_feat == None, self.feature_default_value, pos_feat)
        neg_feat = np.where(neg_feat == None, self.feature_default_value, neg_feat)
        
        return seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat, timestamp, action_type

    def __len__(self):
        """
        返回数据集长度，即用户数量

        Returns:
            usernum: 用户数量
        """
        return len(self.seq_offsets)

    def _init_feat_info(self):
        """
        初始化特征信息, 包括特征缺省值和特征类型

        Returns:
            feat_default_value: 特征缺省值，每个元素为字典，key为特征ID，value为特征缺省值
            feat_types: 特征类型，key为特征类型名称，value为包含的特征ID列表
        """
        feat_default_value = {}
        feat_statistics = {}
        feat_types = {}
        feat_types['user_sparse'] = ['103', '104', '105', '109' ]
        feat_types['item_sparse'] = [
            '100',
            '117',
            '118',
            '101',
            '102',
            '119',
            '120',
            '114',
            '112',
            '121',
            '115',
            '122',
            '116',
        ]
        feat_types['item_array'] = []
        feat_types['user_array'] = ['106', '107', '108', '110']
        feat_types['item_emb'] = self.mm_emb_ids
        feat_types['user_continual'] = []
        feat_types['item_continual'] = []

        for feat_id in feat_types['user_sparse']:
            feat_default_value[feat_id] = 0
            if feat_id == 'weekday':
                feat_statistics[feat_id] = 7
            elif feat_id == 'hour':
                feat_statistics[feat_id] = 24
            elif feat_id == 'time_gap':
                feat_statistics[feat_id] = 15   # 0~19 桶
            else:
                feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['item_sparse']:
            feat_default_value[feat_id] = 0
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['item_array']:
            feat_default_value[feat_id] = [0]
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['user_array']:
            feat_default_value[feat_id] = [0]
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['user_continual']:
            feat_default_value[feat_id] = 0
        for feat_id in feat_types['item_continual']:
            feat_default_value[feat_id] = 0
        for feat_id in feat_types['item_emb']:
            feat_default_value[feat_id] = np.zeros(
                list(self.mm_emb_dict[feat_id].values())[0].shape[0], dtype=np.float32
            )
        return feat_default_value, feat_types, feat_statistics

    def get_time_diff_bin(self, time_diff):
        """
        根据时间差获取对应的分桶索引
        
        Args:
            time_diff: 时间差（秒）
            
        Returns:
            bin_index: 分桶索引 (0-19)
        """
        # 根据你的分析结果定义分桶边界
        bin_edges = [30, 60, 180, 300, 600, 900, 1800, 3600, 10800, 
                    21600, 43200, 86400, 172800, 345600, 604800]
        
        # 找到对应的桶索引
        if time_diff == 0:
            return len(bin_edges) - 1
        for i in range(len(bin_edges) - 1):
            if time_diff > bin_edges[i] and time_diff <= bin_edges[i + 1]:
                return i
        
        # 如果time_diff大于等于最后一个边界，返回最后一个桶
        return len(bin_edges) - 2
        
    def fill_missing_feat(self, feat, item_id):
        """
        对于原始数据中缺失的特征进行填充缺省值

        Args:
            feat: 特征字典
            item_id: 物品ID

        Returns:
            filled_feat: 填充后的特征字典
        """
        if feat == None:
            feat = {}
        filled_feat = {}
        for k in feat.keys():
            filled_feat[k] = feat[k]

        all_feat_ids = []
        for feat_type in self.feature_types.values():
            all_feat_ids.extend(feat_type)
        missing_fields = set(all_feat_ids) - set(feat.keys())
        for feat_id in missing_fields:
            filled_feat[feat_id] = self.feature_default_value[feat_id]
        for feat_id in self.feature_types['item_emb']:
            if item_id != 0 and self.indexer_i_rev[item_id] in self.mm_emb_dict[feat_id]:
                if type(self.mm_emb_dict[feat_id][self.indexer_i_rev[item_id]]) == np.ndarray:
                    filled_feat[feat_id] = self.mm_emb_dict[feat_id][self.indexer_i_rev[item_id]]

        return filled_feat

    def collate_fn(self, batch):
        seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat, timestamp, action_type = zip(*batch)

        seq = torch.from_numpy(np.array(seq))
        pos = torch.from_numpy(np.array(pos))
        neg = torch.from_numpy(np.array(neg))
        token_type = torch.from_numpy(np.array(token_type))
        next_token_type = torch.from_numpy(np.array(next_token_type))
        next_action_type = torch.from_numpy(np.array(next_action_type))

        feature_types = self.feature_types   # 直接用实例里的
        mm_emb_ids = self.mm_emb_ids         # 直接用实例里的
 
        action_type = torch.from_numpy(np.array(action_type))
        # ========== 时间特征处理 ==========
        timestamps = [np.array(ts, dtype=np.int64) for ts in timestamp]
        max_len = max(len(ts) for ts in timestamps)

        # 补齐 padding
        ts_array = np.zeros((len(timestamps), max_len), dtype=np.int64)
        for i, ts in enumerate(timestamps):
            ts_array[i, :len(ts)] = ts
        ts_tensor = torch.from_numpy(ts_array)   # [B, L]

        # week/hour
        week_feat = (ts_tensor // 86400 + 4) % 7      # epoch天数+偏移，转星期 (0–6)
        hour_feat = (ts_tensor % 86400) // 3600       # 秒数转小时 (0–23)
        is_weekend = (week_feat >= 5).long()
        day_of_year = ((ts_tensor // 86400) % 365).long()
        # 一年中的第几周 (0-51)
        week_of_year = ((ts_tensor // 86400) // 7 % 52).long()

        # time delta 分桶
        time_deltas = torch.zeros_like(ts_tensor)
        for b in range(ts_tensor.size(0)):
            for t in range(1, ts_tensor.size(1)):
                diff = ts_tensor[b, t] - ts_tensor[b, t - 1]
                diff = max(int(diff.item()), 0)
                time_deltas[b, t] = self.get_time_diff_bin(diff)

        time_feat = {
            "weekday": week_feat,        # [B, L]
            "is_weekend": is_weekend,    # [B, L]
            "hour": hour_feat,           # [B, L]
            "day_of_year": day_of_year,  # [B, L]
            "week_of_year": week_of_year,# [B, L]
            "time_gap": time_deltas,     # [B, L]
            "action_type": action_type   # [B, L]
        }

        # ========== 内联 feat2tensor ==========
        def feat2tensor(seq_feature, k):
            batch_size = len(seq_feature)

            # ---------- array 特征 ----------
            if k in feature_types['item_array'] or k in feature_types['user_array']:
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

                return torch.from_numpy(batch_data)

            # ---------- 普通 or mm_emb 特征 ----------
            else:
                max_seq_len = max(len(seq_feature[i]) for i in range(batch_size))
                if k in mm_emb_ids:  # 多模态特征
                    feat_dim = len(seq_feature[0][0][k])
                    batch_data = np.zeros((batch_size, max_seq_len, feat_dim), dtype=np.float32)
                    for i in range(batch_size):
                        seq_data = [item[k] for item in seq_feature[i]]
                        for j, val in enumerate(seq_data):
                            batch_data[i, j, :] = val
                else:  # 普通稀疏特征
                    batch_data = np.zeros((batch_size, max_seq_len), dtype=np.int64)
                    for i in range(batch_size):
                        seq_data = [item[k] for item in seq_feature[i]]
                        batch_data[i, :len(seq_data)] = seq_data

                return torch.from_numpy(batch_data)

        def convert_all_features(seq_feat_list):
            feat_dict = {}
            for feat_type, feat_ids in feature_types.items():
                # print(feature_types.items())
                for fid in feat_ids:
                    feat_dict[fid] = feat2tensor(seq_feat_list, fid)
            return feat_dict

        seq_feat = convert_all_features(seq_feat)
        pos_feat = convert_all_features(pos_feat)
        neg_feat = convert_all_features(neg_feat)

        return seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat, time_feat

class MyTestDataset(MyDataset):
    """
    测试数据集
    """

    def __init__(self, data_dir, args):
        self.data_dir = Path(data_dir)
        self._load_data_and_offsets() # 只加载偏移量
        self.maxlen = args.maxlen
        self.mm_emb_ids = args.mm_emb_id

        self.item_feat_dict = json.load(open(Path(data_dir, "item_feat_dict.json"), 'r'))
        self.mm_emb_dict = load_mm_emb(Path(data_dir, "creative_emb"), self.mm_emb_ids)
        with open(self.data_dir / 'indexer.pkl', 'rb') as ff:
            indexer = pickle.load(ff)
            self.itemnum = len(indexer['i'])
            self.usernum = len(indexer['u'])
        self.indexer_i_rev = {v: k for k, v in indexer['i'].items()}
        self.indexer_u_rev = {v: k for k, v in indexer['u'].items()}
        self.indexer = indexer

        self.feature_default_value, self.feature_types, self.feat_statistics = self._init_feat_info()

        # 文件路径，而不是直接打开
        self.data_file_path = self.data_dir / "predict_seq.jsonl"

    def _load_data_and_offsets(self):
        with open(Path(self.data_dir, 'predict_seq_offsets.pkl'), 'rb') as f:
            self.seq_offsets = pickle.load(f)

    def _process_cold_start_feat(self, feat):
        """
        处理冷启动特征。训练集未出现过的特征value为字符串，默认转换为0.可设计替换为更好的方法。
        """
        processed_feat = {}
        for feat_id, feat_value in feat.items():
            if type(feat_value) == list:
                value_list = []
                for v in feat_value:
                    if type(v) == str:
                        value_list.append(0)
                    else:
                        value_list.append(v)
                processed_feat[feat_id] = value_list
            elif type(feat_value) == str:
                processed_feat[feat_id] = 0
            else:
                processed_feat[feat_id] = feat_value
        return processed_feat

    def __getitem__(self, uid):
        """
        获取单个用户的数据，并进行padding处理，生成模型需要的数据格式

        Args:
            uid: 用户在self.data_file中储存的行号
        Returns:
            seq: 用户序列ID
            token_type: 用户序列类型，1表示item，2表示user
            seq_feat: 用户序列特征，每个元素为字典，key为特征ID，value为特征值
            user_id: user_id eg. user_xxxxxx ,便于后面对照答案
        """
        user_sequence = self._load_user_data(uid)  # 动态加载用户数据

        ext_user_sequence = []
        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, act_type, ts_now = record_tuple
            if u:
                if type(u) == str:  # 如果是字符串，说明是user_id
                    user_id = u
                else:  # 如果是int，说明是re_id
                    user_id = self.indexer_u_rev[u]
            if u and user_feat:
                if type(u) == str:
                    u = 0
                if user_feat:
                    user_feat = self._process_cold_start_feat(user_feat)
                ext_user_sequence.insert(0, (u, user_feat, 2, 0))

            if i and item_feat:
                # 序列对于训练时没见过的item，不会直接赋0，而是保留creative_id，creative_id远大于训练时的itemnum
                if i > self.itemnum:
                    i = 0
                if item_feat:
                    item_feat = self._process_cold_start_feat(item_feat)
                ext_user_sequence.append((i, item_feat, 1, ts_now or 0))

        seq = np.zeros([self.maxlen + 1], dtype=np.int32)
        token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        seq_feat = np.empty([self.maxlen + 1], dtype=object)
        timestamp = np.empty([self.maxlen + 1], dtype=np.int64)
        action_type = np.zeros([self.maxlen + 1], dtype=np.int32)

        idx = self.maxlen

        ts = set()
        for record_tuple in ext_user_sequence:
            if record_tuple[2] == 1 and record_tuple[0]:
                ts.add(record_tuple[0])

        for record_tuple in reversed(ext_user_sequence):
            i, feat, type_, ts_now = record_tuple
            feat = self.fill_missing_feat(feat, i)
           
            
            seq[idx] = i
            token_type[idx] = type_
            if act_type is not None:
                action_type[idx] = act_type
            seq_feat[idx] = feat
            timestamp[idx] = ts_now
            
            nxt = record_tuple
            idx -= 1
            if idx == -1:
                break

        seq_feat = np.where(seq_feat == None, self.feature_default_value, seq_feat)

        return seq, token_type, seq_feat, user_id, timestamp, action_type

    def __len__(self):
        """
        Returns:
            len(self.seq_offsets): 用户数量
        """
        with open(Path(self.data_dir, 'predict_seq_offsets.pkl'), 'rb') as f:
            temp = pickle.load(f)
        return len(temp)

    def collate_fn(self, batch):
        seq, token_type, seq_feat, user_id, timestamp, action_type = zip(*batch)

        seq = torch.from_numpy(np.array(seq))
        token_type = torch.from_numpy(np.array(token_type))

        feature_types = self.feature_types   # 直接用实例里的
        mm_emb_ids = self.mm_emb_ids         # 直接用实例里的

        action_type = torch.from_numpy(np.array(action_type))
        expose_items = []
        seq_np = np.array(seq)   # [B, L]
        act_np = np.array(action_type)
        for i in range(len(user_id)):
            # 取出当前用户序列中 action_type==0 且 item_id>0 的位置
            exposed = seq_np[i][(act_np[i] == 0) & (seq_np[i] > 0)]
            expose_items.append(set(exposed.tolist()))
        # ========== 时间特征处理 ==========
        timestamps = [np.array(ts, dtype=np.int64) for ts in timestamp]
        max_len = max(len(ts) for ts in timestamps)

        # 补齐 padding
        ts_array = np.zeros((len(timestamps), max_len), dtype=np.int64)
        for i, ts in enumerate(timestamps):
            ts_array[i, :len(ts)] = ts
        ts_tensor = torch.from_numpy(ts_array)   # [B, L]

        # week/hour
        week_feat = (ts_tensor // 86400 + 4) % 7      # epoch天数+偏移，转星期 (0–6)
        hour_feat = (ts_tensor % 86400) // 3600       # 秒数转小时 (0–23)
        is_weekend = (week_feat >= 5).long()
        day_of_year = ((ts_tensor // 86400) % 365).long()
        # 一年中的第几周 (0-51)
        week_of_year = ((ts_tensor // 86400) // 7 % 52).long()

        # time delta 分桶
        time_deltas = torch.zeros_like(ts_tensor)
        for b in range(ts_tensor.size(0)):
            for t in range(1, ts_tensor.size(1)):
                diff = ts_tensor[b, t] - ts_tensor[b, t - 1]
                diff = max(int(diff.item()), 0)
                time_deltas[b, t] = self.get_time_diff_bin(diff)

        time_feat = {
            "weekday": week_feat,        # [B, L]
            "is_weekend": is_weekend,    # [B, L]
            "hour": hour_feat,           # [B, L]
            "day_of_year": day_of_year,  # [B, L]
            "week_of_year": week_of_year,# [B, L]
            "time_gap": time_deltas,     # [B, L]
            "action_type": action_type   # [B, L]
        }

        # ========== 内联 feat2tensor ==========
        def feat2tensor(seq_feature, k):
            batch_size = len(seq_feature)

            # ---------- array 特征 ----------
            if k in feature_types['item_array'] or k in feature_types['user_array']:
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

                return torch.from_numpy(batch_data)

            # ---------- 普通 or mm_emb 特征 ----------
            else:
                max_seq_len = max(len(seq_feature[i]) for i in range(batch_size))
                if k in mm_emb_ids:  # 多模态特征
                    feat_dim = len(seq_feature[0][0][k])
                    batch_data = np.zeros((batch_size, max_seq_len, feat_dim), dtype=np.float32)
                    for i in range(batch_size):
                        seq_data = [item[k] for item in seq_feature[i]]
                        for j, val in enumerate(seq_data):
                            batch_data[i, j, :] = val
                else:  # 普通稀疏特征
                    batch_data = np.zeros((batch_size, max_seq_len), dtype=np.int64)
                    for i in range(batch_size):
                        seq_data = [item[k] for item in seq_feature[i]]
                        batch_data[i, :len(seq_data)] = seq_data

                return torch.from_numpy(batch_data)

        def convert_all_features(seq_feat_list):
            feat_dict = {}
            for feat_type, feat_ids in feature_types.items():
                for fid in feat_ids:
                    feat_dict[fid] = feat2tensor(seq_feat_list, fid)
            return feat_dict

        seq_feat = convert_all_features(seq_feat)

        return seq, token_type, seq_feat, user_id, time_feat, expose_items


def save_emb(emb, save_path):
    """
    将Embedding保存为二进制文件

    Args:
        emb: 要保存的Embedding，形状为 [num_points, num_dimensions]
        save_path: 保存路径
    """
    num_points = emb.shape[0]  # 数据点数量
    num_dimensions = emb.shape[1]  # 向量的维度
    print(f'saving {save_path}')
    with open(Path(save_path), 'wb') as f:
        f.write(struct.pack('II', num_points, num_dimensions))
        emb.tofile(f)


def load_mm_emb(mm_path, feat_ids):
    """
    加载多模态特征Embedding

    Args:
        mm_path: 多模态特征Embedding路径
        feat_ids: 要加载的多模态特征ID列表

    Returns:
        mm_emb_dict: 多模态特征Embedding字典，key为特征ID，value为特征Embedding字典（key为item ID，value为Embedding）
    """
    SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
    mm_emb_dict = {}
    for feat_id in tqdm(feat_ids, desc='Loading mm_emb'):
        shape = SHAPE_DICT[feat_id]
        emb_dict = {}
        if feat_id != '81':
            try:
                base_path = Path(mm_path, f'emb_{feat_id}_{shape}')
                for json_file in base_path.glob('*.json'):
                    with open(json_file, 'r', encoding='utf-8') as file:
                        for line in file:
                            data_dict_origin = json.loads(line.strip())
                            insert_emb = data_dict_origin['emb']
                            if isinstance(insert_emb, list):
                                insert_emb = np.array(insert_emb, dtype=np.float32)
                            data_dict = {data_dict_origin['anonymous_cid']: insert_emb}
                            emb_dict.update(data_dict)
            except Exception as e:
                print(f"transfer error: {e}")
        if feat_id == '81':
            with open(Path(mm_path, f'emb_{feat_id}_{shape}.pkl'), 'rb') as f:
                emb_dict = pickle.load(f)
        mm_emb_dict[feat_id] = emb_dict
        print(f'Loaded #{feat_id} mm_emb')
    return mm_emb_dict