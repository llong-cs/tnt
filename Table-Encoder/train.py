import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

# parallel (multi-gpu) training
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist

import os
import argparse
import wandb
from dotenv import load_dotenv
import itertools
from datetime import datetime

from data_process.dataset import TableDataset
from torch.utils.data import DataLoader
from model import TableEncoder
from utils import *

from torch.cuda.amp import autocast, GradScaler

load_dotenv()
DATASETS_PATH = os.environ["DATASETS_PATH"]
MODELS_PATH = os.environ["MODELS_PATH"]
SENTENCE_TRANSFORMER = os.environ["SENTENCE_TRANSFORMER"]

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str)
parser.add_argument("--from_csv", action="store_true", default=False)
parser.add_argument("--log_activate", action="store_true", default=False)
parser.add_argument("--log_interval", type=int, default=10)
parser.add_argument("--eval_interval", type=int, default=150)
parser.add_argument("--save_path", type=str, default="checkpoints")
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--pin_memory", type=bool, default=True)
parser.add_argument("--drop_last", type=bool, default=False)
parser.add_argument("--comment", type=str, default=None)

# model param
# parser.add_argument('--num_columns', default=50, type=int)
parser.add_argument('--transformer_depth', default=12, type=int)
# hidden size of the model = 16 * 64 = 1024
parser.add_argument('--attention_heads', default=16, type=int)
parser.add_argument('--dim_head', default=64, type=int)
parser.add_argument("--pooling", default="mean", type=str, choices=["cls", "mean"])
parser.add_argument('--attention_dropout', default=0.1, type=float)
parser.add_argument('--ff_dropout', default=0.1, type=float)
parser.add_argument("--load_model", type=bool, default=False)
parser.add_argument("--model_path", type=str, default=None)
parser.add_argument("--col_name", action="store_true", default=False)
parser.add_argument("--numeric_mlp", action="store_true", default=False)

# training param
parser.add_argument("--pred_type", type=str, default="contrastive")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--optimizer", default="AdamW", type=str, choices=["AdamW", "Adam", "SGD"])
parser.add_argument("--scheduler", default="cosine", type=str, choices=["cosine", "linear"])
parser.add_argument("--st_lr", type=float, default=1e-5)
parser.add_argument("--enc_lr", type=float, default=1e-5)
parser.add_argument("--warmup_steps", type=int, default=0) 

parser.add_argument("--gradient_checkpoint", action="store_true", default=False)
parser.add_argument("--gradient_cache", action="store_true", default=False)
parser.add_argument("--chunk_size", type=int, default=1024) # a large number to avoid minibatch

def ddp_setup(rank, world_size):
   """
   Args:
       rank: Unique identifier of each process
       world_size: Total number of processes
   """
   os.environ["MASTER_ADDR"] = "localhost"
   os.environ["MASTER_PORT"] = "12563"
   dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
   torch.cuda.set_device(rank)
    
class Trainer:
    def __init__(self, args, model=None):
        self.args = args
        self.model = model
        
        # initialize optimizer
        if args.pred_type == "contrastive":
            params = itertools.chain(self.model.module.transformer.parameters(), self.model.module.col_specific_projection_head.parameters())
        if args.col_name:
            params = itertools.chain(params, self.model.module.col_name_projection_head.parameters())
        if args.pooling == 'cls':
            params = itertools.chain(params, self.model.module.cls.parameters())
        if args.numeric_mlp:
            params = itertools.chain(params, self.model.module.num_mlp.parameters())
        
        if args.optimizer == "SGD":
            self.optimizer = optim.SGD(
                params=params, lr=args.lr, momentum=0.9, weight_decay=5e-4
            )
            from utils import get_scheduler
            self.scheduler = get_scheduler(args, self.optimizer)
        elif args.optimizer == "Adam":
            self.optimizer = optim.Adam(params=params, lr=args.lr)
        elif args.optimizer == "AdamW":
            self.optimizer = optim.AdamW(params=params, lr=args.enc_lr, weight_decay=0.01)
            
        # set loss function
        if args.pred_type == "contrastive":
            # each criterion for contrastive learning is responsible for maintaining its own dictionary (MoCo)
            self.criterion = ContLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
            
        self.wamred_up = False
                    
    def run(self):
        args = self.args
                
        # test: no warmup, only 1 epoch, more frequent evaluation
        if args.comment == "test":
            self.args.eval_interval = args.log_interval * 2
            warmup_dataloader, train_dataloader, eval_dataloader = self.prepare_data(shuffle=False)
            self.unfreeze_st()
            train_dataloader.sampler.set_epoch(1)
            self.train(train_dataloader, eval_dataloader, 1)
            return
        
        # prepare data
        warmup_dataloader, train_dataloader, eval_dataloader = self.prepare_data()
        
        # warmup with 1w samples
        self.train(warmup_dataloader, None, 0)
             
        # save warmed-up model
        if args.rank == 0:
            current_date = datetime.now()
            formatted_date = current_date.strftime("%Y%m%d")
            model_name = f'{args.pred_type}-{SENTENCE_TRANSFORMER}-{args.seed}-{args.model_path}-{args.st_lr}-{args.enc_lr}-{args.batch_size}-{args.comment}-{formatted_date}'
            save_path = f"{args.save_path}/{model_name}"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(self.model.module.state_dict(), os.path.join(save_path, f"model_warmup.pt"))
        
        # train the model with 8w samples for {args.epochs} epochs
        for epoch in range(args.epochs):
            train_dataloader.sampler.set_epoch(epoch + 1)
            self.train(train_dataloader, eval_dataloader, epoch + 1)
        
    def prepare_data(self, shuffle=True):
        args = self.args
        
        warmup_data_path = os.path.join(args.data_path, "warmup_data.np")
        warmup_dataset = TableDataset(data_path=warmup_data_path, pred_type=args.pred_type, idx=None, model=self.model, from_csv=args.from_csv, numeric_mlp=args.numeric_mlp)
        warmup_sampler = DistributedSampler(
            dataset=warmup_dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=shuffle,
        )
        warmup_dataloader = DataLoader(warmup_dataset, batch_size=args.batch_size, sampler=warmup_sampler, num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=args.drop_last)
        
        official_data_path = os.path.join(args.data_path, "training_data.np")
        with open(official_data_path, "rb") as f:
            train_data = np.load(f)
            # randomly split the dataset into train and valid
            # the number of valid samples is 5% of the whole dataset
            data_num = len(train_data)
            valid_num = data_num // 20
            perm_idx = np.random.permutation(data_num)
            train_idx = perm_idx[: -valid_num]
            valid_idx = perm_idx[-valid_num:]
            
        train_dataset = TableDataset(data_path=official_data_path, pred_type=args.pred_type, idx=train_idx, model=self.model, from_csv=args.from_csv, numeric_mlp=args.numeric_mlp)
        train_sampler = DistributedSampler(
            dataset=train_dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=shuffle,
        )
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=args.drop_last)
        
        eval_dataset = TableDataset(data_path=official_data_path, pred_type=args.pred_type, idx=valid_idx, model=self.model, from_csv=args.from_csv, numeric_mlp=args.numeric_mlp)
        eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=args.drop_last, shuffle=shuffle)
        
        if args.rank == 0:
            print("Data prepared.")
            print(f"Warmup dataset size: {len(warmup_dataset)} | Train dataset size: {len(train_dataset)} | Eval dataset size: {len(eval_dataset)} | Total dataset size: {len(warmup_dataset) + len(train_dataset) + len(eval_dataset)}")
        
        self.args.warmup_steps = len(warmup_dataset) // 2 // args.world_size // args.batch_size
        
        return warmup_dataloader, train_dataloader, eval_dataloader
    
    def unfreeze_st(self):
        args = self.args
        self.model.module.unfreeze_st()
        if args.from_csv:
            self.optimizer.add_param_group({'params': self.model.module.st.encoder.parameters(), 'lr': args.st_lr})
        if SENTENCE_TRANSFORMER == "puff-base-v1":
            self.optimizer.add_param_group({'params': self.model.module.vector_linear.parameters(), 'lr': args.st_lr})
        self.wamred_up = True
        
        
    def train(self, dataloader, eval_dataloader, epoch):    
        args = self.args
                
        self.model.train()

        running_loss = []
        
        if args.pred_type == "name_prediction":
            for (
                anchor_table,
                shuffled_table,
                anchor_table_mask,
                shuffled_table_mask,
                target,
                target_mask,
            ) in dataloader:
                self.optimizer.zero_grad()
                (
                    anchor_table,
                    shuffled_table,
                    anchor_table_mask,
                    shuffled_table_mask,
                    target,
                    target_mask,
                ) = (
                    anchor_table.float().cuda(),
                    shuffled_table.float().cuda(),
                    anchor_table_mask.float().cuda(),
                    shuffled_table_mask.float().cuda(),
                    target.float().cuda(),
                    target_mask.int().cuda()
                )

                # Predict with model
                output = self.model(anchor_table, anchor_table_mask, shuffled_table, shuffled_table_mask)
                
                # output size = [bs, 100, 384]
                # target size = [bs, 50, 384]
                # target_mask size = [bs, 50], marks the shuffled tokens with 1
                output_flatten = output[:, 50:, :].reshape(-1, output.shape[-1])
                target_flatten = target.reshape(-1, output.shape[-1])
                target_mask_flatten = target_mask.reshape(-1)
                pred = output_flatten[torch.where(target_mask_flatten == 1)]
                gt = target_flatten[torch.where(target_mask_flatten == 1)]
                
                loss = self.criterion(pred, gt)
                
                if args.log_activate:
                    wandb.log({"batch_loss": loss})
                else:
                    print(loss)
                
                # for output_, row_, target_ in zip(output, shuffled_table[:,0,:,:], target):
                #     loss += criterion(output_[target_[np.where(target_ >= 0)[0]] + anchor_table.shape[1]], row_[torch.where(target_ >= 0)])

                # Calculate loss
                # loss = criterion(output, target)

                loss.backward()
                self.optimizer.step()
                if args.optimizer == "SGD":
                    self.scheduler.step()

                running_loss.append(loss.item())
        else:
            scaler = GradScaler()

            for batch_idx, (
                anchor_table_input_ids,
                anchor_table_attention_mask,
                anchor_table_token_type_ids,
                shuffled_table_input_ids,
                shuffled_table_attention_mask,
                shuffled_table_token_type_ids,
                anchor_table_mask,
                shuffled_table_mask,
                shuffle_idx
            ) in enumerate(dataloader):
                if epoch == 0:
                    step = batch_idx + epoch * len(dataloader)
                else:
                    step = batch_idx + (epoch - 1) * len(dataloader)
                
                (
                    anchor_table_input_ids,
                    anchor_table_attention_mask,
                    anchor_table_token_type_ids,
                    shuffled_table_input_ids,
                    shuffled_table_attention_mask,
                    shuffled_table_token_type_ids,
                    anchor_table_mask,
                    shuffled_table_mask,
                    shuffle_idx
                ) = (
                    anchor_table_input_ids.cuda(),
                    anchor_table_attention_mask.cuda(),
                    anchor_table_token_type_ids.cuda(),
                    shuffled_table_input_ids.cuda(),
                    shuffled_table_attention_mask.cuda(),
                    shuffled_table_token_type_ids.cuda(),
                    anchor_table_mask.cuda(),
                    shuffled_table_mask.cuda(),
                    shuffle_idx.int().cuda()
                )
                
                index_ = torch.arange(shuffle_idx.size(1)).repeat(shuffle_idx.size(0), 1).to(shuffle_idx.device).to(shuffle_idx.dtype)
                shuffle_idx = torch.where(shuffle_idx < 0, index_, shuffle_idx)
                shuffle_idx_expanded = shuffle_idx.unsqueeze(-1).expand(-1, -1, self.model.module.cont_dim).to(torch.int64)
                mask_flatten = anchor_table_mask[:,0,:].detach() # will be flatten later

                self.optimizer.zero_grad()

                if not args.gradient_cache:
                    with autocast():
                        if SENTENCE_TRANSFORMER == "puff-base-v1":
                            anchor_emb_col_spe = self.model(anchor_table_input_ids, anchor_table_attention_mask, None, anchor_table_mask)
                            shuffled_emb_col_spe = self.model(shuffled_table_input_ids, shuffled_table_attention_mask, None, shuffled_table_mask)
                        elif SENTENCE_TRANSFORMER == "all-MiniLM-L6-v2" or SENTENCE_TRANSFORMER == 'bge-small-en-v1.5':
                            anchor_emb_col_spe = self.model(anchor_table_input_ids, anchor_table_attention_mask, anchor_table_token_type_ids, anchor_table_mask)
                            shuffled_emb_col_spe = self.model(shuffled_table_input_ids, shuffled_table_attention_mask, shuffled_table_token_type_ids, shuffled_table_mask)

                        # if args.col_name:
                        #     col_name_emb = F.normalize(self.model.module.col_name_projection_head(args.col_name), p=2, dim=-1)

                        # re-order (align with the shuffled table)
                        anchor_emb_col_spe = anchor_emb_col_spe.gather(1, shuffle_idx_expanded)

                        # if args.col_name:
                        #     col_name_emb = col_name_emb.gather(1, shuffle_idx_expanded) 
                        
                        # [bs, 100, 256] -> [n, 256]
                        anchor_emb_col_spe_flatten = anchor_emb_col_spe.reshape(-1, anchor_emb_col_spe.shape[-1])[torch.where(mask_flatten.reshape(-1) == 1)[0],:]
                        shuffle_emb_col_spe_flatten = shuffled_emb_col_spe.reshape(-1, shuffled_emb_col_spe.shape[-1])[torch.where(mask_flatten.reshape(-1) == 1)[0],:]
                        loss = self.criterion(q=anchor_emb_col_spe_flatten, k=torch.cat([shuffle_emb_col_spe_flatten,anchor_emb_col_spe_flatten], dim=0), mask=anchor_table_mask, target="col")
                    
                    # loss.backward()   
                    scaler.scale(loss).backward()
                    
                else:
                    # Gradient Cache
                    # 1. forward no grad
                    with torch.no_grad(), autocast(torch.bfloat16):
                        if SENTENCE_TRANSFORMER == "puff-base-v1":
                            anchor_emb_col_spe = self.model(anchor_table_input_ids, anchor_table_attention_mask, None, anchor_table_mask)
                            shuffled_emb_col_spe = self.model(shuffled_table_input_ids, shuffled_table_attention_mask, None, shuffled_table_mask)
                        elif SENTENCE_TRANSFORMER == "all-MiniLM-L6-v2" or SENTENCE_TRANSFORMER == 'bge-small-en-v1.5':
                            anchor_emb_col_spe = self.model(anchor_table_input_ids, anchor_table_attention_mask, anchor_table_token_type_ids, anchor_table_mask)
                            shuffled_emb_col_spe = self.model(shuffled_table_input_ids, shuffled_table_attention_mask, shuffled_table_token_type_ids, shuffled_table_mask)
                        anchor_emb_col_spe = anchor_emb_col_spe.gather(1, shuffle_idx_expanded)
                        anchor_emb_col_spe_flatten = anchor_emb_col_spe.reshape(-1, anchor_emb_col_spe.shape[-1])[torch.where(mask_flatten.reshape(-1) == 1)[0],:]
                        shuffle_emb_col_spe_flatten = shuffled_emb_col_spe.reshape(-1, shuffled_emb_col_spe.shape[-1])[torch.where(mask_flatten.reshape(-1) == 1)[0],:]
                        
                    # 2. compute loss and get grad
                    anchor_emb_col_spe_flatten = anchor_emb_col_spe_flatten.detach().requires_grad_()
                    shuffle_emb_col_spe_flatten = shuffle_emb_col_spe_flatten.detach().requires_grad_()
                    with autocast(torch.bfloat16):
                        loss = self.criterion(q=anchor_emb_col_spe_flatten, k=torch.cat([shuffle_emb_col_spe_flatten,anchor_emb_col_spe_flatten], dim=0), mask=anchor_table_mask, target="col")
                    scaler.scale(loss).backward()
                    loss = loss.detach()
                    anchor_emb_col_spe_flatten_grad = anchor_emb_col_spe_flatten.grad
                    shuffle_emb_col_spe_flatten_grad = shuffle_emb_col_spe_flatten.grad

                    # 3. forward grad
                    batch_size = anchor_table_input_ids.shape[0]
                    chunk_size = args.chunk_size
                    num_chunk = (batch_size + chunk_size - 1) // chunk_size

                    s_g = 0
                    e_g = 0
                    for i in range(num_chunk):
                        s = i * chunk_size
                        e = min(s + chunk_size, batch_size)

                        with autocast(torch.bfloat16):
                            if SENTENCE_TRANSFORMER == "puff-base-v1":
                                anchor_emb_col_spe = self.model(anchor_table_input_ids[s:e], anchor_table_attention_mask[s:e], None, anchor_table_mask[s:e])
                            elif SENTENCE_TRANSFORMER == "all-MiniLM-L6-v2" or SENTENCE_TRANSFORMER == 'bge-small-en-v1.5':
                                anchor_emb_col_spe = self.model(anchor_table_input_ids[s:e], anchor_table_attention_mask[s:e], anchor_table_token_type_ids[s:e], anchor_table_mask[s:e])
                            anchor_emb_col_spe = anchor_emb_col_spe.gather(1, shuffle_idx_expanded[s:e])
                            anchor_emb_col_spe_flatten = anchor_emb_col_spe.reshape(-1, anchor_emb_col_spe.shape[-1])[torch.where(mask_flatten[s:e].reshape(-1) == 1)[0],:]
                            e_g += anchor_emb_col_spe_flatten.shape[0]
                        anchor_grad = torch.dot(anchor_emb_col_spe_flatten_grad[s_g:e_g].flatten(), anchor_emb_col_spe_flatten.flatten())
                        scaler.scale(anchor_grad).backward()

                        with autocast(torch.bfloat16):
                            if SENTENCE_TRANSFORMER == "puff-base-v1":
                                shuffled_emb_col_spe = self.model(shuffled_table_input_ids[s:e], shuffled_table_attention_mask[s:e], None, shuffled_table_mask[s:e])
                            elif SENTENCE_TRANSFORMER == "all-MiniLM-L6-v2" or SENTENCE_TRANSFORMER == 'bge-small-en-v1.5':
                                shuffled_emb_col_spe = self.model(shuffled_table_input_ids[s:e], shuffled_table_attention_mask[s:e], shuffled_table_token_type_ids[s:e], shuffled_table_mask[s:e])
                            shuffle_emb_col_spe_flatten = shuffled_emb_col_spe.reshape(-1, shuffled_emb_col_spe.shape[-1])[torch.where(mask_flatten[s:e].reshape(-1) == 1)[0],:]
                        shuffle_grad = torch.dot(shuffle_emb_col_spe_flatten_grad[s_g:e_g].flatten(), shuffle_emb_col_spe_flatten.flatten())
                        scaler.scale(shuffle_grad).backward()

                        s_g = e_g
            
                # self.optimizer.step()
                scaler.step(self.optimizer)
                if args.optimizer == "SGD":
                    self.scheduler.step()
                scaler.update()

                running_loss.append(loss.item())
                
                if step >= args.warmup_steps and not self.wamred_up:
                    self.unfreeze_st()
                
                if args.rank == 0:
                    if step % args.log_interval == 0:
                        if epoch == 0:
                            print(f"============== Warmup Step-{step // args.log_interval} ==============")
                        else:
                            print(f"============== Step-{step // args.log_interval} ==============")
                            
                        avg_loss = torch.mean(torch.tensor(running_loss))
                        running_loss = []
                        if args.log_activate:
                            wandb.log({"train_loss": avg_loss})
                        print(f"Train loss: {avg_loss}")
                        
                if epoch > 0 and step % args.eval_interval == 0:
                    
                    self.eval(eval_dataloader)
                    
                    if args.rank == 0:
                        current_date = datetime.now()
                        formatted_date = current_date.strftime("%Y%m%d")
                        model_name = f'{args.pred_type}-{SENTENCE_TRANSFORMER}-{args.seed}-{args.model_path}-{args.st_lr}-{args.enc_lr}-{args.batch_size}-{args.comment}-{formatted_date}'
                        save_path = f"{args.save_path}/{model_name}"
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        torch.save(self.model.module.state_dict(), os.path.join(save_path, f"model_{step}.pt"))
                
            
            # if args.rank == 0:
            #     print(f"============== Epoch-{epoch} ==============")
            #     avg_loss = torch.mean(torch.tensor(running_loss))
            #     running_loss = []
            #     if args.log_activate:
            #         wandb.log({"train_loss": avg_loss})
            #     else:
            #         print(f"Train loss: {avg_loss}")

    def eval(self, dataloader):    
        args = self.args
        
        self.model.eval()
        
        running_loss = []
                    
        if args.pred_type == "name_prediction":
            for (
                anchor_table,
                shuffled_table,
                anchor_table_mask,
                shuffled_table_mask,
                target,
                target_mask,
            ) in dataloader:
                (
                    anchor_table,
                    shuffled_table,
                    anchor_table_mask,
                    shuffled_table_mask,
                    target,
                    target_mask,
                ) = (
                    anchor_table.float().cuda(),
                    shuffled_table.float().cuda(),
                    anchor_table_mask.float().cuda(),
                    shuffled_table_mask.float().cuda(),
                    target.float().cuda(),
                    target_mask.int().cuda()
                )

                # Predict with model
                output = self.model(anchor_table, anchor_table_mask, shuffled_table, shuffled_table_mask)
                
                output_flatten = output[:, 50:, :].reshape(-1, output.shape[-1])
                target_flatten = target.reshape(-1, output.shape[-1])
                target_mask_flatten = target_mask.reshape(-1)
                pred = output_flatten[torch.where(target_mask_flatten == 1)]
                gt = target_flatten[torch.where(target_mask_flatten == 1)]
                
                loss = self.criterion(pred, gt)
                
                running_loss.append(loss.item())
        else:
            for batch_idx, (
                anchor_table_input_ids,
                anchor_table_attention_mask,
                anchor_table_token_type_ids,
                shuffled_table_input_ids,
                shuffled_table_attention_mask,
                shuffled_table_token_type_ids,
                anchor_table_mask,
                shuffled_table_mask,
                shuffle_idx
            ) in enumerate(dataloader):
                
                (
                    anchor_table_input_ids,
                    anchor_table_attention_mask,
                    anchor_table_token_type_ids,
                    shuffled_table_input_ids,
                    shuffled_table_attention_mask,
                    shuffled_table_token_type_ids,
                    anchor_table_mask,
                    shuffled_table_mask,
                    shuffle_idx
                ) = (
                    anchor_table_input_ids.cuda(),
                    anchor_table_attention_mask.cuda(),
                    anchor_table_token_type_ids.cuda(),
                    shuffled_table_input_ids.cuda(),
                    shuffled_table_attention_mask.cuda(),
                    shuffled_table_token_type_ids.cuda(),
                    anchor_table_mask.cuda(),
                    shuffled_table_mask.cuda(),
                    shuffle_idx.int().cuda()
                )
                
                with torch.no_grad():
                    if SENTENCE_TRANSFORMER == "puff-base-v1":
                        anchor_emb_col_spe = self.model(anchor_table_input_ids, anchor_table_attention_mask, None, anchor_table_mask)
                        shuffled_emb_col_spe = self.model(shuffled_table_input_ids, shuffled_table_attention_mask, None, shuffled_table_mask)
                    elif SENTENCE_TRANSFORMER == "all-MiniLM-L6-v2" or SENTENCE_TRANSFORMER == 'bge-small-en-v1.5':
                        anchor_emb_col_spe = self.model(anchor_table_input_ids, anchor_table_attention_mask, anchor_table_token_type_ids, anchor_table_mask)
                        shuffled_emb_col_spe = self.model(shuffled_table_input_ids, shuffled_table_attention_mask, shuffled_table_token_type_ids, shuffled_table_mask)
                        
                    if args.col_name:
                        col_name_emb = F.normalize(self.model.module.col_name_projection_head(col_name), p=2, dim=-1)
                                        
                # re-order (align with the shuffled table)
                index_ = torch.arange(shuffle_idx.size(1)).repeat(shuffle_idx.size(0), 1).to(shuffle_idx.device).to(shuffle_idx.dtype)
                shuffle_idx = torch.where(shuffle_idx < 0, index_, shuffle_idx)
                shuffle_idx_expanded = shuffle_idx.unsqueeze(-1).expand(-1, -1, anchor_emb_col_spe.size(2)).to(torch.int64)
                anchor_emb_col_spe = anchor_emb_col_spe.gather(1, shuffle_idx_expanded)
                if args.col_name:
                    col_name_emb = col_name_emb.gather(1, shuffle_idx_expanded) 
                
                mask_flatten = anchor_table_mask[:,0,:].reshape(-1).detach()    
                
                # [bs, 100, 256] -> [n, 256]
                anchor_emb_col_spe_flatten = anchor_emb_col_spe.reshape(-1, anchor_emb_col_spe.shape[-1])[torch.where(mask_flatten == 1)[0],:]
                shuffle_emb_col_spe_flatten = shuffled_emb_col_spe.reshape(-1, shuffled_emb_col_spe.shape[-1])[torch.where(mask_flatten == 1)[0],:]
                loss = self.criterion(q=anchor_emb_col_spe_flatten, k=torch.cat([shuffle_emb_col_spe_flatten,anchor_emb_col_spe_flatten], dim=0), mask=anchor_table_mask, target="col")
                
                running_loss.append(loss.item())
                
                if batch_idx >= 10:
                    break
                                
        eval_loss = torch.mean(torch.tensor(running_loss))
        
        if args.rank == 0:
            if args.log_activate:
                wandb.log({"eval_loss": eval_loss})
            else:
                print(f"Eval loss: {eval_loss}")
    
def main(rank, world_size, args):    
    # Setup DDP
    ddp_setup(rank, world_size)
    
    # update args
    args.rank = rank
    args.world_size = world_size
    args.eval_interval = args.log_interval * 20
        
    # initialize model
    model = TableEncoder(
        num_cols=args.num_columns,
        depth=args.transformer_depth,
        heads=args.attention_heads,
        attn_dropout=args.attention_dropout,
        ff_dropout=args.ff_dropout,
        attentiontype="colrow",
        pred_type=args.pred_type,
        dim_head=args.dim_head,
        pooling=args.pooling,
        col_name=args.col_name,
        numeric_mlp=args.numeric_mlp,
        gradient_checkpoint=args.gradient_checkpoint
    )

    if args.load_model:
        model.load_state_dict(torch.load(args.model_path))
        print(f"Model loaded from {args.model_path}.")
        args.lr = 1e-5
        
    model = model.cuda()
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    
    args.embedding_size = model.module.dim
    
    if rank == 0:
        print('Model created.')
        print(get_parameter_number(model))
    
        # log to wandb
        if args.log_activate:
            current_date = datetime.now()
            formatted_date = current_date.strftime("%Y%m%d")
            project_name = f'{args.pred_type}-{SENTENCE_TRANSFORMER}-{args.seed}-{args.model_path}-{args.st_lr}-{args.enc_lr}-{args.batch_size}-{args.comment}-{formatted_date}'
            wandb.init(project="table-encoder", name=project_name)
            wandb.config.update(args)
    
    trainer = Trainer(args, model)
    trainer.run()

    if rank == 0:
        if args.log_activate:
            wandb.finish()
    
    # Clean up
    destroy_process_group()


if __name__ == "__main__":
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    world_size = torch.cuda.device_count()
    print(f"Let's use {world_size} GPUs and {mp.cpu_count()} CPUs!")
    mp.spawn(main, args=(world_size, args, ), nprocs=world_size, join=True)
