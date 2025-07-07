import torch
import numpy as np
import torch.nn as nn
import torch.distributed as dist
from torch.nn import functional as F


def get_scheduler(args, optimizer):
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    elif args.scheduler == "linear":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[args.epochs // 2.667, args.epochs // 1.6, args.epochs // 1.142],
            gamma=0.1,
        )
    return scheduler

class ContLoss(nn.Module):
    def __init__(
        self,
        queue_len=512,
        temperature=0.07
    ):
        super().__init__()
        self.temperature = temperature
        self.dim = 256
        self.queue_len = queue_len
        
        # create the queue (for contrastive learning)
        self.register_buffer("queue", torch.randn(queue_len, self.dim))
        self.queue = nn.functional.normalize(self.queue, dim=1).cuda()
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        # filter out zero padding
        keys = keys[torch.norm(keys, dim=1) > 1e-8]
        
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr[0])

        # replace the keys at ptr (dequeue and enqueue)
        # self.queue[:, ptr : ptr + batch_size] = keys.T
        if ptr + batch_size <= self.queue_len:
            self.queue[ptr : ptr + batch_size] = keys
        else:
            self.queue[ptr : self.queue_len] = keys[: self.queue_len - ptr]
            self.queue[: batch_size - (self.queue_len - ptr)] = keys[self.queue_len - ptr :]
        
        ptr = (ptr + batch_size) % self.queue_len  # move pointer

        self.queue_ptr[0] = ptr
                                    
    # q/k size: (bs, 100, 256)
    # mask size: (bs, 50, 100)
    def forward(self, q, k, mask=None, target='col'):
        # for semantic table
        # q = col_emb, k = col_name_emb
        # for non-semantic table
        # q,k = col_emb

        batch_size = q.shape[0]
                
        # k_dict = torch.cat([k, self.queue], dim=0)
        # k_dict = k
    
        logits = torch.einsum("nc,kc->nk", [q, k]) / self.temperature    
                
        positive_mask = torch.zeros_like(logits, dtype=torch.float)
        positive_mask = (
            torch.scatter(
                positive_mask,
                1,
                torch.arange(batch_size).view(-1, 1).cuda(),
                1,
            )
            .detach()
        )

        # get table mask
        # col_num = torch.sum(mask[:,0,:], dim=1).int()            
        # table_mask = torch.zeros_like(logits, dtype=torch.float)
        
        # accu = 0
        # for col in col_num:
        #     table_mask[accu:accu+col, accu:accu+col] = 1
        #     accu += col
        
        # assert accu == batch_size
        
        # if table_mask.shape[1] > batch_size:
        #     table_mask[:, batch_size:] = table_mask[:, :batch_size]
        
        # table_mask = table_mask.detach()
        
        self_mask = torch.ones_like(logits, dtype=torch.float)    
        if k.shape[0] > batch_size:
            self_mask[:, batch_size:] -= torch.eye(batch_size).cuda()
        self_mask = self_mask.detach()
            
        logits_exp = torch.exp(logits)
        # logits_exp_sum = torch.sum(logits_exp * self_mask * table_mask, dim=1, keepdim=True)
        logits_exp_sum = torch.sum(logits_exp * self_mask, dim=1, keepdim=True)

        loss = -torch.sum(
            torch.log(logits_exp / logits_exp_sum) * positive_mask, dim=1
        ).mean()
        
        # dequeue and enqueue
        # self.dequeue_and_enqueue(k.clone().detach())
        
        return loss
    
@torch.no_grad()
def concat_all_gather(tensor, size=100):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient and can only gather tensors of consistent sizes.
    """
    
    tensor = F.pad(tensor, (0, 0, 0, size - tensor.shape[0]), 'constant', 0)
    # print(tensor.shape)
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(dist.get_world_size())
    ]
    # print(tensor.shape)
    dist.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # param_info = {'Total': total_num, 'Trainable': trainable_num}
    param_info = f"Total params: {total_num} | Trainable params: {trainable_num}"
    return param_info

def count_param_size(param):
    memory_size = param.element_size() * param.numel()
    memory_size_GB = memory_size / (1024 ** 3)
    return memory_size_GB

def get_cuda_state():
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        
        total_memory = torch.cuda.get_device_properties(current_device).total_memory
        reserved_memory = torch.cuda.memory_reserved(current_device)
        allocated_memory = torch.cuda.memory_allocated(current_device)
        free_memory = -allocated_memory + reserved_memory
        
        print(f"Total: {total_memory / (1024 ** 3):.2f} GB | Reserved: {reserved_memory / (1024 ** 3):.2f} GB | Allocated: {allocated_memory / (1024 ** 3):.2f} GB | Free: {free_memory / (1024 ** 3):.2f} GB")
    else:
        print("GPU is not available.")

def get_opt_state(optimizer):
    for i, param_group in enumerate(optimizer.param_groups):
        print(f"Param group {i}, Learning rate: {param_group['lr']}")
        
def get_grad(module):
    for name, param in module.named_parameters():
        if param.grad is not None:
            print(f"Gradient of {name} in fc: {param.grad.norm().item()}")
