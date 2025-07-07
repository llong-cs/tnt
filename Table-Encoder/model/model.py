import torch
import torch.nn.functional as F
from torch import nn, einsum
import numpy as np
from einops import rearrange
from torch.utils.checkpoint import checkpoint

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def ff_encodings(x,B):
    x_proj = (2. * np.pi * x.unsqueeze(-1)) @ B.t()
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

# classes

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# attention

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x, **kwargs):
        return self.net(x)

class RowColAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 16,
        dropout = 0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        # s = batch size
        # b = number of rows
        # h = number of heads
        # n = number of columns
        q, k, v = map(lambda t: rearrange(t, 's b n (h d) -> s b h n d', h = h), (q, k, v))
        sim = einsum('s b h i d, s b h j d -> s b h i j', q, k) * self.scale

        # masking
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1).repeat(1, sim.shape[1], sim.shape[2], 1, 1)
            sim = sim.masked_fill(mask == 0, torch.finfo(sim.dtype).min)

        attn = sim.softmax(dim = -1)
        out = einsum('s b h i j, s b h j d -> s b h i d', attn, v)
        out = rearrange(out, 's b h n d -> s b n (h d)', h = h)
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=16, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        # s = batch size
        # b = number of rows
        # h = number of heads
        # n = number of columns
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v)
        )
        
        sim = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        # masking
        # torch.Size([12, 300, 300])
        if mask is not None:
            mask = (
                mask.unsqueeze(1)
                .repeat(1, sim.shape[1], 1, 1)
            )
            # print(mask.shape)
            # print(mask)
            sim = sim.masked_fill(mask == 0, torch.finfo(sim.dtype).min)

        attn = sim.softmax(dim=-1)
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)", h=h)
        return self.to_out(out)

class Qformer(nn.Module):
    def __init__(self, dim, dim_head, inner_dim, query_num):
        super().__init__()
        
        self.heads = inner_dim // dim_head
        self.scale = dim_head**-0.5
        self.q = nn.Parameter(torch.randn(query_num, inner_dim))
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.ff = PreNorm(inner_dim, Residual(FeedForward(inner_dim)))
    
    def forward(self, x, mask=None):
        x = rearrange(x, 's b n d -> s n b d')
        
        h = self.heads
        k, v = self.to_kv(x).chunk(2, dim=-1)
        q = self.q.unsqueeze(0).unsqueeze(0).repeat(x.shape[0], x.shape[1], 1, 1)
        # s = batch size
        # b = number of rows
        # h = number of heads
        # n = number of columns
        q, k, v = map(lambda t: rearrange(t, 's b n (h d) -> s b h n d', h = h), (q, k, v))
        sim = einsum('s b h i d, s b h j d -> s b h i j', q, k) * self.scale

        # masking
        if mask is not None:
            mask = rearrange(mask, 's i j -> s j i')
            mask = mask[:,0,:].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, sim.shape[1], sim.shape[2], sim.shape[3], 1)
            sim = sim.masked_fill(mask == 0, torch.finfo(sim.dtype).min)

        attn = sim.softmax(dim = -1)
        out = einsum('s b h i j, s b h j d -> s b h i d', attn, v)
        out = rearrange(out, 's b h n d -> s b n (h d)', h = h)
        
        out = self.ff(out)
        return out

class RowColTransformer(nn.Module):
    # dim = dim of each token
    # nfeats = number of features (columns)
    # depth = number of attention layers
    # heads = number of heads in multihead attention
    # dim_head = dim of each head
    def __init__(self, dim, nfeats, depth, heads, dim_head, attn_dropout, ff_dropout, style='col', mask=None, gradient_checkpoint=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.style = style
        self.gradient_checkpoint = gradient_checkpoint

        for _ in range(depth):
            if self.style == 'colrow':
                # self.layers.append(nn.ModuleList([
                #     PreNorm(dim, Residual(Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout))),
                #     PreNorm(dim, Residual(FeedForward(dim, dropout = ff_dropout))),
                #     PreNorm(dim*nfeats, Residual(Attention(dim*nfeats, heads = heads, dim_head = 64, dropout = attn_dropout))),
                #     PreNorm(dim*nfeats, Residual(FeedForward(dim*nfeats, dropout = ff_dropout))),
                # ]))
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, Residual(RowColAttention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout))),
                    PreNorm(dim, Residual(FeedForward(dim, dropout = ff_dropout))),
                    PreNorm(dim, Residual(RowColAttention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout))),
                    PreNorm(dim, Residual(FeedForward(dim, dropout = ff_dropout))),
                ]))
            else:
                self.layers.append(nn.ModuleList([
                    PreNorm(dim*nfeats, Residual(Attention(dim*nfeats, heads = heads, dim_head = 64, dropout = attn_dropout))),
                    PreNorm(dim*nfeats, Residual(FeedForward(dim*nfeats, dropout = ff_dropout))),
                ]))

    def forward(self, x, mask=None):
        _, _, n, _ = x.shape # [bs, n_rows, n_cols, dim]
        row_mask = None
        col_mask = None
        if mask is not None:
            # row_mask = torch.zeros(mask.shape[0], mask.shape[1], mask.shape[1]).to(mask.device)
            # col_mask = torch.zeros(mask.shape[0], mask.shape[2], mask.shape[2]).to(mask.device)
            # for i in range(mask.shape[0]):
            #     row_mask[i] = torch.matmul(mask[i], mask[i].T)
            #     col_mask[i] = torch.matmul(mask[i].T, mask[i])
            col_mask = einsum('b i j, b i k -> b j k', mask, mask)
            row_mask = einsum('b i j, b k j -> b i k', mask, mask)
        # print(col_mask.shape, row_mask.shape)
        if self.style == 'colrow':
            for attn1, ff1, attn2, ff2 in self.layers: 
                if self.gradient_checkpoint:
                    x = checkpoint(attn1, x, mask=col_mask, use_reentrant=False)
                    x = checkpoint(ff1, x, use_reentrant=False)
                else:
                    x = attn1(x, mask=col_mask)
                    x = ff1(x)
                x = rearrange(x, 's b n d -> s n b d')
                if self.gradient_checkpoint:
                    x = checkpoint(attn2, x, mask=row_mask, use_reentrant=False)
                    x = checkpoint(ff2, x, use_reentrant=False)
                else:
                    x = attn2(x, mask=row_mask)
                    x = ff2(x)
                x = rearrange(x, 's n b d -> s b n d', n = n)
        else:
            for attn1, ff1 in self.layers:
                x = rearrange(x, 's b n d -> s 1 b (n d)')
                if self.gradient_checkpoint:
                    x = checkpoint(attn1, x, use_reentrant=False)
                    x = checkpoint(ff1, x, use_reentrant=False)
                else:
                    x = attn1(x)
                    x = ff1(x)
                x = rearrange(x, 's 1 b (n d) -> s b n d', n = n)
        return x


# transformer
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, attn_dropout, ff_dropout, gradient_checkpoint=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.gradient_checkpoint = gradient_checkpoint

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Residual(Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout))),
                PreNorm(dim, Residual(FeedForward(dim, dropout = ff_dropout))),
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            if self.gradient_checkpoint:
                x = checkpoint(attn, x, mask=mask, use_reentrant=False)
                x = checkpoint(ff, x, use_reentrant=False)
            else:
                x = attn(x, mask=mask)
                x = ff(x)
        return x


# mlp
class MLP(nn.Module):
    def __init__(self, dims, act = None):
        super().__init__()
        dims_pairs = list(zip(dims[:-1], dims[1:]))
        layers = []
        for ind, (dim_in, dim_out) in enumerate(dims_pairs):
            is_last = ind >= (len(dims) - 1)
            linear = nn.Linear(dim_in, dim_out)
            layers.append(linear)

            if is_last:
                continue
            if act is not None:
                layers.append(act)

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class simple_MLP(nn.Module):
    def __init__(self, dims, gradient_checkpoint=False):
        super(simple_MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(dims[0], dims[1]),
            nn.ReLU(),
            nn.Linear(dims[1], dims[2])
        )
        self.gradient_checkpoint = gradient_checkpoint
        
    def forward(self, x):
        if len(x.shape)==1:
            x = x.view(x.size(0), -1)
        if self.gradient_checkpoint:
            x = checkpoint(self.layers, x, use_reentrant=False)
        else:
            x = self.layers(x)
        return x

# main class

class TabAttention(nn.Module):
    def __init__(
        self,
        *,
        categories,
        num_continuous,
        dim,
        depth,
        heads,
        dim_head = 16,
        dim_out = 1,
        mlp_hidden_mults = (4, 2),
        mlp_act = None,
        num_special_tokens = 1,
        continuous_mean_std = None,
        attn_dropout = 0.,
        ff_dropout = 0.,
        lastmlp_dropout = 0.,
        cont_embeddings = 'MLP',
        scalingfactor = 10,
        attentiontype = 'col'
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'

        # categories related calculations
        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        # create category embeddings table

        self.num_special_tokens = num_special_tokens
        self.total_tokens = self.num_unique_categories + num_special_tokens

        # for automatically offsetting unique category ids to the correct position in the categories embedding table
        categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value = num_special_tokens)
        categories_offset = categories_offset.cumsum(dim = -1)[:-1]
        
        self.register_buffer('categories_offset', categories_offset)


        self.norm = nn.LayerNorm(num_continuous)
        self.num_continuous = num_continuous
        self.dim = dim
        self.cont_embeddings = cont_embeddings
        self.attentiontype = attentiontype

        if self.cont_embeddings == 'MLP':
            self.simple_MLP = nn.ModuleList([simple_MLP([1,100,self.dim]) for _ in range(self.num_continuous)])
            input_size = (dim * self.num_categories)  + (dim * num_continuous)
            nfeats = self.num_categories + num_continuous
        else:
            print('Continous features are not passed through attention')
            input_size = (dim * self.num_categories) + num_continuous
            nfeats = self.num_categories 

        # transformer
        if attentiontype == 'col':
            self.transformer = Transformer(
                num_tokens = self.total_tokens,
                dim = dim,
                depth = depth,
                heads = heads,
                dim_head = dim_head,
                attn_dropout = attn_dropout,
                ff_dropout = ff_dropout
            )
        elif attentiontype in ['row','colrow'] :
            self.transformer = RowColTransformer(
                num_tokens = self.total_tokens,
                dim = dim,
                nfeats= nfeats,
                depth = depth,
                heads = heads,
                dim_head = dim_head,
                attn_dropout = attn_dropout,
                ff_dropout = ff_dropout,
                style = attentiontype
            )

        l = input_size // 8
        hidden_dimensions = list(map(lambda t: l * t, mlp_hidden_mults))
        all_dimensions = [input_size, *hidden_dimensions, dim_out]
        
        self.mlp = MLP(all_dimensions, act = mlp_act)
        self.embeds = nn.Embedding(self.total_tokens, self.dim) #.to(device)

        cat_mask_offset = F.pad(torch.Tensor(self.num_categories).fill_(2).type(torch.int8), (1, 0), value = 0) 
        cat_mask_offset = cat_mask_offset.cumsum(dim = -1)[:-1]

        con_mask_offset = F.pad(torch.Tensor(self.num_continuous).fill_(2).type(torch.int8), (1, 0), value = 0) 
        con_mask_offset = con_mask_offset.cumsum(dim = -1)[:-1]

        self.register_buffer('cat_mask_offset', cat_mask_offset)
        self.register_buffer('con_mask_offset', con_mask_offset)

        self.mask_embeds_cat = nn.Embedding(self.num_categories*2, self.dim)
        self.mask_embeds_cont = nn.Embedding(self.num_continuous*2, self.dim)

    def forward(self, x_categ, x_cont,x_categ_enc,x_cont_enc):
        device = x_categ.device
        if self.attentiontype == 'justmlp':
            if x_categ.shape[-1] > 0:
                flat_categ = x_categ.flatten(1).to(device)
                x = torch.cat((flat_categ, x_cont.flatten(1).to(device)), dim = -1)
            else:
                x = x_cont.clone()
        else:
            if self.cont_embeddings == 'MLP':
                x = self.transformer(x_categ_enc,x_cont_enc.to(device))
            else:
                if x_categ.shape[-1] <= 0:
                    x = x_cont.clone()
                else: 
                    flat_categ = self.transformer(x_categ_enc).flatten(1)
                    x = torch.cat((flat_categ, x_cont), dim = -1)                    
        flat_x = x.flatten(1)
        return self.mlp(flat_x)

class TableDecoder(nn.Module):
    def __init__(self, dim, nfeats, depth, heads, dim_head, attn_dropout, ff_dropout, style='col'):
        super().__init__()
        self.encoder = RowColTransformer(
            dim=dim,
            nfeats=nfeats,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            style=style
        )
        self.decoder = Transformer(
            dim=dim,
            # nfeats=nfeats,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
        )
        

    def forward(self, anchor_table_emb, shuffled_table, mask = None, anchor_mask = None):
        # 1. Encode shuffled table -> shuffled_table_emb: [bs, 2*n_cols, dim]
        # 2. Concat anchor_table_emb and shuffled_table_emb -> emb: [bs, 3*n_cols, dim]
        # 3. Segment emb?
        # 4. Decode emb with common transformer -> output: [bs, 3*n_cols, dim]
        shuffled_table_emb = self.encoder(shuffled_table, mask=mask) # [bs, n_rows, 2*n_cols, dim]
        
        shuffled_table_emb_ = get_flatten_table_emb(shuffled_table_emb, mask)
        
        # torch.Size([12, 50, 100]) torch.Size([12, 5, 200]) -> torch.Size([12, 1, 300])
        joint_mask = torch.cat((anchor_mask[:,0,:], mask[:,0,:]), dim=1).unsqueeze(1)
        assert torch.sum(joint_mask) == torch.sum(anchor_mask[:,0,:]) + torch.sum(mask[:,0,:])
        # print(joint_mask.shape)
        # torch.Size([12, 300, 300])
        joint_mask = einsum('b i j, b i k -> b j k', joint_mask, joint_mask)
        # print(joint_mask)
        
        emb = torch.cat((anchor_table_emb, shuffled_table_emb_), dim=1)
        output = self.decoder(emb, joint_mask)
        return output

# table_emb size: [bs, n_rows, n_cols, dim]
# mask size: [bs, n_rows, n_cols]
# output size: [bs, n_cols, dim]
def get_flatten_table_emb(table_emb, mask):
    flatten_table_emb = torch.zeros(table_emb.size(0), table_emb.size(2), table_emb.size(3)).to(table_emb.device)
    row_num = torch.sum(mask, dim=1).int()
    for i in range(len(table_emb)):
        flatten_table_emb[i] = torch.mean(table_emb[i, :row_num[i,0], :, :], dim=0)
    return flatten_table_emb