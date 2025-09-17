import argparse
import json
import os
import struct
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MyTestDataset, save_emb
from model import BaselineModel
import random

def set_seed(seed=3407):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果有多个GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_ckpt_path():
    ckpt_path = os.environ.get("MODEL_OUTPUT_PATH")
    if ckpt_path is None:
        raise ValueError("MODEL_OUTPUT_PATH is not set")
    for item in os.listdir(ckpt_path):
        if item.endswith(".pt"):
            return os.path.join(ckpt_path, item)


def get_args():
    parser = argparse.ArgumentParser()

    # Train params
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=101, type=int)

    # Baseline Model construction
    parser.add_argument('--hidden_units', default=512, type=int)
    parser.add_argument('--embedding_dim', default=128, type=int)
    parser.add_argument('--num_blocks', default=8, type=int)
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--l2_emb', default=1e-4, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--norm_first', action='store_true')

    # MMemb Feature ID
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, choices=[str(s) for s in range(81, 87)])

    args = parser.parse_args()

    return args


def read_result_ids(file_path):
    with open(file_path, 'rb') as f:
        # Read the header (num_points_query and FLAGS_query_ann_top_k)
        num_points_query = struct.unpack('I', f.read(4))[0]  # uint32_t -> 4 bytes
        query_ann_top_k = struct.unpack('I', f.read(4))[0]  # uint32_t -> 4 bytes

        print(f"num_points_query: {num_points_query}, query_ann_top_k: {query_ann_top_k}")

        # Calculate how many result_ids there are (num_points_query * query_ann_top_k)
        num_result_ids = num_points_query * query_ann_top_k

        # Read result_ids (uint64_t, 8 bytes per value)
        result_ids = np.fromfile(f, dtype=np.uint64, count=num_result_ids)

        return result_ids.reshape((num_points_query, query_ann_top_k))


def process_cold_start_feat(feat):
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
# 假设 process_cold_start_feat 已经定义

def get_candidate_emb(indexer, feat_types, feat_default_value, mm_emb_dict, model):
    """
    生产候选库item的id和embedding

    Args:
        indexer: 索引字典
        feat_types: 特征类型，分为user和item的sparse, array, emb, continual类型
        feature_default_value: 特征缺省值
        mm_emb_dict: 多模态特征字典
        model: 模型
    Returns:
        retrieve_id2creative_id: 索引id->creative_id的dict
    """
    EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
    candidate_path = Path(os.environ.get('EVAL_DATA_PATH'), 'predict_set.jsonl')
    item_ids, creative_ids, retrieval_ids, features = [], [], [], []
    retrieve_id2creative_id = {}

    with open(candidate_path, 'r') as f:
        for line in f:
            line = json.loads(line)
            # 读取item特征，并补充缺失值
            feature = line['features']
            creative_id = line['creative_id']
            retrieval_id = line['retrieval_id']
            item_id = indexer[creative_id] if creative_id in indexer else 0
            missing_fields = set(
                feat_types['item_sparse'] + feat_types['item_array'] + feat_types['item_continual']
            ) - set(feature.keys())
            feature = process_cold_start_feat(feature)
            for feat_id in missing_fields:
                feature[feat_id] = feat_default_value[feat_id]
            for feat_id in feat_types['item_emb']:
                if creative_id in mm_emb_dict[feat_id]:
                    feature[feat_id] = mm_emb_dict[feat_id][creative_id]
                else:
                    feature[feat_id] = np.zeros(EMB_SHAPE_DICT[feat_id], dtype=np.float32)
            item_ids.append(item_id)
            creative_ids.append(creative_id)
            retrieval_ids.append(retrieval_id)
            features.append(feature)
            retrieve_id2creative_id[retrieval_id] = creative_id

    # 保存候选库的embedding和sid
    candidate_embs = model.save_item_emb(item_ids, retrieval_ids, features, os.environ.get('EVAL_RESULT_PATH'))
    with open(Path(os.environ.get('EVAL_RESULT_PATH'), "retrive_id2creative_id.json"), "w") as f:
        json.dump(retrieve_id2creative_id, f)
    return candidate_embs, retrieve_id2creative_id

def infer():
    """
    使用矩阵乘法替代Faiss的Top-K检索，并在推理时剔除
    用户交互中曝光但未点击的 item。
    """
    args = get_args()
    set_seed(3407)
    data_path = os.environ.get('EVAL_DATA_PATH')

    # --- 1. 数据和模型加载 ---
    test_dataset = MyTestDataset(data_path, args)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=12,
        collate_fn=test_dataset.collate_fn
    )

    usernum, itemnum = test_dataset.usernum, test_dataset.itemnum
    feat_statistics, feat_types = test_dataset.feat_statistics, test_dataset.feature_types

    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)
    model.eval()

    ckpt_path = get_ckpt_path()
    model.load_state_dict(torch.load(ckpt_path, map_location=torch.device(args.device)))

    # --- 2. 生成候选库Embedding ---
    candidate_embs, retrieve_id2creative_id = get_candidate_emb(
        test_dataset.indexer['i'],
        test_dataset.feature_types,
        test_dataset.feature_default_value,
        test_dataset.mm_emb_dict,
        model,
    )
    candidate_embs = candidate_embs.to(args.device)

    # 建立 creative_id -> candidate_embs 行索引映射
    retrieval_ids = list(retrieve_id2creative_id.keys())
    creative_ids = [retrieve_id2creative_id[rid] for rid in retrieval_ids]
    creative_id2idx = {cid: idx for idx, cid in enumerate(creative_ids)}

    # --- 3. 生成 Query Embedding & 曝光未点击集合 ---
    all_query_embs = []
    user_list = []
    all_expose_items = []   # 每个用户曝光未点击 creative_id 集合

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="生成 Query Embeddings"):
            # ⚠️ collate_fn 需返回 expose_items
            seq, token_type, seq_feat, user_id, time_feat, expose_items = batch
            seq = seq.to(args.device)

            query_embs = model.predict(seq, seq_feat, token_type, time_feat)

            all_query_embs.append(query_embs.cpu())
            user_list.extend(user_id)
            all_expose_items.extend(expose_items)

    all_query_embs = torch.cat(all_query_embs, dim=0)

    # --- 4. 矩阵乘法检索 + 剔除曝光未点击 item ---
    all_top10_indices = []
    query_batch_size = 1024  # 可按显存调整

    with torch.no_grad():
        for i in tqdm(range(0, all_query_embs.size(0), query_batch_size), desc="矩阵乘法检索"):
            query_batch = all_query_embs[i:i + query_batch_size].to(args.device)
            scores = torch.matmul(query_batch, candidate_embs.T)  # [B, num_candidates]

            # ✅ 对每个用户剔除曝光未点击 item
            for row_idx, user_idx in enumerate(range(i, i + query_batch.size(0))):
                filter_items = all_expose_items[user_idx]
                if not filter_items:
                    continue
                ban_indices = [
                    creative_id2idx[cid]
                    for cid in filter_items
                    if cid in creative_id2idx
                ]
                if ban_indices:
                    scores[row_idx, torch.tensor(ban_indices, device=scores.device)] = float('-inf')

            _, topk_indices = torch.topk(scores, k=10, dim=1)
            all_top10_indices.append(topk_indices.cpu())

    final_topk_indices = torch.cat(all_top10_indices, dim=0)

    # --- 5. ID 转换 ---
    top10s = []
    for user_indices in tqdm(final_topk_indices, desc="转换ID"):
        user_top10 = []
        for item_index in user_indices:
            retrieved_id = retrieval_ids[item_index.item()]
            creative_id = retrieve_id2creative_id.get(retrieved_id, 0)
            user_top10.append(creative_id)
        top10s.append(user_top10)

    print(f"检索完成，共处理 {len(user_list)} 个用户。")
    return top10s, user_list