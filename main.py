import argparse
import json
import os
import time
from pathlib import Path
import bitsandbytes as bnb
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import random

from dataset import MyDataset
from model import BaselineModel
from transformers import get_cosine_schedule_with_warmup


def set_seed(seed=3407):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_args():
    parser = argparse.ArgumentParser()

    # Train params
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', default=0.0008, type=float)
    parser.add_argument('--maxlen', default=101, type=int)
    parser.add_argument('--grad_clip_norm', default=1.0, type=float)
    parser.add_argument('--warmup_ratio', default=0.1, type=float, help='warmup learning rate ratio')
    parser.add_argument('--temp', default=0.04, type=float)
    # Baseline Model construction
    parser.add_argument('--hidden_units', default=64, type=int)
    parser.add_argument('--embedding_dim', default=64, type=int)
    parser.add_argument('--num_blocks', default=8, type=int)
    parser.add_argument('--num_epochs', default=8, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--dropout_rate', default=0.0, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--norm_first', action='store_true')
    parser.add_argument('--resume', default=None, type=str, help='checkpoint 路径(用于恢复训练)')

    # MMemb Feature ID
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, choices=[str(s) for s in range(81, 87)])
    return parser.parse_args()


def save_checkpoint(save_dir, epoch, global_step, model, optimizer, scheduler):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / f"epoch{epoch}_step{global_step}.pt"
    torch.save({
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }, ckpt_path)
    print(f"✅ 保存 checkpoint 到: {ckpt_path}")


def load_checkpoint(ckpt_path, model, optimizer, scheduler, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    print(f"✅ 从 {ckpt_path} 恢复: epoch={ckpt['epoch']} step={ckpt['global_step']}")
    return ckpt["epoch"] + 1, ckpt["global_step"]  # 下一轮从 epoch+1 开始


if __name__ == '__main__':
    Path(os.environ.get('TRAIN_LOG_PATH')).mkdir(parents=True, exist_ok=True)
    Path(os.environ.get('TRAIN_TF_EVENTS_PATH')).mkdir(parents=True, exist_ok=True)
    Path(os.environ.get('TRAIN_CKPT_PATH')).mkdir(parents=True, exist_ok=True)

    log_file = open(Path(os.environ.get('TRAIN_LOG_PATH'), 'train.log'), 'a')
    writer = SummaryWriter(os.environ.get('TRAIN_TF_EVENTS_PATH'))

    args = get_args()
    set_seed(42)

    # === 数据集 ===
    data_path = os.environ.get('TRAIN_DATA_PATH')
    dataset = MyDataset(data_path, args)
    train_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=12, collate_fn=dataset.collate_fn
    )
    usernum, itemnum = dataset.usernum, dataset.itemnum
    feat_statistics, feat_types = dataset.feat_statistics, dataset.feature_types

    # === 模型 ===
    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)
    for name, param in model.named_parameters():
        if "user_emb" in name or "item_emb" in name:
            torch.nn.init.zeros_(param.data)
        else:
            try:
                torch.nn.init.xavier_normal_(param.data)
            except:
                pass
    model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0
    model.user_emb.weight.data[0, :] = 0
    for k in model.sparse_emb:
        model.sparse_emb[k].weight.data[0, :] = 0

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2_emb)
    total_steps = len(train_loader) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # === 恢复训练(如果有) ===
    epoch_start = 1
    global_step = 0
    if args.resume is not None and os.path.exists(args.resume):
        epoch_start, global_step = load_checkpoint(args.resume, model, optimizer, scheduler, args.device)

    print("🚀 Start training")
    for epoch in range(epoch_start, args.num_epochs + 1):
        model.train()
        if args.inference_only:
            break

        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat, time_feat = batch
            seq, pos, neg = seq.to(args.device), pos.to(args.device), neg.to(args.device)
            print(seq)
            optimizer.zero_grad()

            # === bf16 混合精度 ===
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pos_logits, neg_logits, anchor_emb, pos_emb, neg_emb = model(
                    seq, pos, neg, token_type, next_token_type, next_action_type,
                    seq_feat, pos_feat, neg_feat, time_feat
                )
                infonce_loss = model.compute_infonce_loss(
                    seq, pos, neg, token_type, next_token_type, next_action_type,
                    seq_feat, pos_feat, neg_feat, writer, time_feat, args.temp
                )
                loss = infonce_loss

            log_json = json.dumps({
                'global_step': global_step,
                'loss': loss.item(),
                'epoch': epoch,
                'time': time.time()
            })
            log_file.write(log_json + '\n')
            log_file.flush()
            writer.add_scalar('Loss/train', loss.item(), global_step)
            print(log_json)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()
            scheduler.step()
            global_step += 1

        # === 每个 epoch 结束后保存 checkpoint ===
        save_checkpoint(os.environ.get('TRAIN_CKPT_PATH'), epoch, global_step, model, optimizer, scheduler)

    print("🎯 Training Done.")
    writer.close()
    log_file.close()