import math
import torch
from einops import rearrange
from torch import Tensor

attn_count = 0

def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor, attn_map_list=None) -> Tensor:
    q, k = apply_rope(q, k, pe)

    use_attn_map = attn_map_list is not None

    if use_attn_map:
        scale = 1 / math.sqrt(q.shape[-1])
        sim = torch.einsum("b h i d, b h j d -> b h i j", q, k) * scale
        del q, k

        """
        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')  
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> b h 1 j', h=h)
            sim.masked_fill_(~mask, max_neg_value)
        """


        sim = sim.softmax(dim=-1)  # softmax over j
        #print("org sim shape", sim.shape)

        _sim = sim.squeeze(-1) # drop 'key' dimension
        _sim = _sim.mean(dim=1) # average over heads
        #print("avg sim shape", _sim.shape)

        _attn_map = _sim.detach().cpu()       

        attn_map_list.append(_attn_map)

        #self._attn_map = _sim.detach().cpu()


        # calculate attention
        x = torch.einsum("b h i j, b h j d -> b h i d", sim, v)  # [b, h, i, d]

    else:
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)


    # Then optionally collapse heads back:
    x = rearrange(x, "B H L D -> B L (H D)")



    return x


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=pos.dtype, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)
