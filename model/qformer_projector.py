import torch
import torch.nn as nn
import re
from torch.nn import functional as F
from torch.nn.init import trunc_normal_
import numpy as np

def get_1d_sincos_pos_embed_from_grid(embed_dim, length):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    pos = np.arange(length)
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    emb = torch.from_numpy(emb).to(dtype = torch.bfloat16, device = 'cuda')
    return emb
class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x
    
class Projector(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size, dim_head, num_queries, mlp_type = 'identity', **kwargs):
        super().__init__()
        self.num_queries = num_queries
        self.dim_head = dim_head
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        
        self.query = nn.Parameter(torch.randn(num_queries, decoder_hidden_size))
        trunc_normal_(self.query, std=.02)
        # self.kv_proj = nn.Linear(encoder_hidden_size, decoder_hidden_size * 2, bias=False)
        self.kv_proj = nn.Linear(encoder_hidden_size, decoder_hidden_size, bias=False)
        self.attn = nn.MultiheadAttention(decoder_hidden_size, num_heads = decoder_hidden_size // dim_head, batch_first=True)
        self.ln_q = nn.LayerNorm(decoder_hidden_size)
        self.ln_kv = nn.LayerNorm(decoder_hidden_size)
        
        if mlp_type == 'identity':
            self.mlp = Identity()
            print('mlp is identity')
        else:
            self.mlp = nn.Sequential(
                nn.Linear(decoder_hidden_size, decoder_hidden_size),
                nn.GELU(),
                nn.Linear(decoder_hidden_size, decoder_hidden_size)
            )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, key_padding_mask=None):
        # x: n_cols, n_rows, encoder_hidden_size
        x = self.kv_proj(x)
        x = self.ln_kv(x)
        # kv_pos_embeds = get_1d_sincos_pos_embed_from_grid(self.decoder_hidden_size, x.shape[-2])
        # kv_pos_embeds.requires_grad_(False)
        k = x
        v = x
        

        q = self.query.unsqueeze(0).repeat(x.shape[0], 1, 1)
        q_pos_embeds = get_1d_sincos_pos_embed_from_grid(self.decoder_hidden_size, self.num_queries)
        q_pos_embeds.requires_grad_(False)
        q = self.ln_q(q) + q_pos_embeds
        # q: n_cols, num_queries, decoder_hidden_size        
        # key_padding_mask: bs, n_cols, n_rows        
        assert k.shape == v.shape, (k.shape, v.shape)
        out = self.attn(q, k, v, key_padding_mask)[0]
        
        out = self.mlp(out)
        return out
    
    
        
        
        