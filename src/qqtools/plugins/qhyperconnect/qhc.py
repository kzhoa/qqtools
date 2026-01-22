"""
# language=python

# --- usage demo 1---
from somewhere import qhyperconnect

hc = qhyperconnect(2)
hc = qhyperconnect(2, agg='sum') # set an agg to all layers, default `None`

class Block(torch.nn.Module):

    def __init__(self):

        self.attn = hc.layer(AttnModule(128))
        self.ffn = hc.layer(FeedForward(128))

    def forward(self, h)

        # we need to expand it explicitly
        hn = hc.expand(h) # (..., d, n)

        hn = self.attn(hn)
        hn = self.ffn(hn)

        # then mix it explicitly
        out = hc.mix(hn, agg='sum')

        return out


# --- usage demo 2---
from somewhere import qhyperconnect

hc = qhyperconnect(2)
hc = qhyperconnect(2, agg='sum') # set an agg to all layers, default `None`

class Block(torch.nn.Module):

    def __init__(self):

        self.attn = hc.closedLayer(AttnModule(128))
        self.ffn = hc.closedLayer(FeedForward(128))

        # Note:
        # if the global `agg` is not set
        # you need to set `agg` of each closed layer

        self.attn = hc.closedLayer(AttnModule(128), agg='sum')
        self.attn = hc.closedLayer(FeedForward(128), agg='mean')

    def forward(self, h)

        # do not need to expand or mix in closed layer
        h = self.attn(h) # (..., d)
        out = self.ffn(h) # (..., d)
        return out



# --- usage demo 3 -----
# fit different feature_dim

# by default hyperc assume the feature_dim == -1
# e.g. h.shape=(..., d)
# this is suitable for most cases,
# for example (bz, d), (bz, seq, d)

# but for cv cases
# h.shape=(bz, d, height, width)
# where feature_dim = 1

# user can specificly set feature_dim

"""

from typing import Union

import torch
import torch.nn.functional as F

__all__ = ["qhyperconnect"]


def qhyperconnect(expansion, dynamic=False, d=None, agg=None, feature_dim=-1) -> "HyperConnectFactory":
    return HyperConnectFactory(expansion=expansion, dynamic=dynamic, d=d, agg=agg, feature_dim=feature_dim)


def move_feature_dim_to_last(
    tensor: torch.Tensor, feature_dim: Union[int, str] = "auto", return_permutes=False
) -> torch.Tensor:
    """
    Example:
        >>> # Image (batch, channels, height, width)
        >>> img = torch.randn(2, 3, 32, 32)
        >>> out = move_feature_dim_to_last(img, 1)
        >>> out.shape  # (2, 32, 32, 3)

        >>> # Text (batch, seq_len, hidden_dim)
        >>> text = torch.randn(4, 10, 768)
        >>> out = move_feature_dim_to_last(text, 2)
        >>> out.shape  # (4, 10, 768)
    """
    if feature_dim == "auto":
        if tensor.dim() == 4:  # (b,c,h,w)
            feature_dim = 1
        elif tensor.dim() == 3:  #  (b,s,d)
            feature_dim = 2
        else:
            raise ValueError("unrecognized")

    if feature_dim < 0:
        feature_dim += tensor.dim()

    if not 0 <= feature_dim < tensor.dim():
        raise ValueError(f"feature_dim {feature_dim} out of range: [0, {tensor.dim()})")

    if feature_dim == tensor.dim() - 1:
        return tensor
    dims = [i for i in range(tensor.dim()) if i != feature_dim] + [feature_dim]
    return tensor.permute(*dims)


def move_last_dim_to_feature(tensor: torch.Tensor, feature_dim: Union[int, str] = "auto") -> torch.Tensor:
    """
    Example:
        >>> # Image (batch, height, width, channels) -> (batch, channels, height, width)
        >>> img = torch.randn(2, 32, 32, 3)
        >>> out = move_last_dim_to_feature(img, 1)
        >>> out.shape  # (2, 3, 32, 32)

        >>> # Text (batch, seq_len, hidden_dim) -> (batch, hidden_dim, seq_len)
        >>> text = torch.randn(4, 10, 768)
        >>> out = move_last_dim_to_feature(text, 1)
        >>> out.shape  # (4, 768, 10)
    """
    if feature_dim == "auto":
        if tensor.dim() == 4:
            feature_dim = 1
        elif tensor.dim() == 3:
            feature_dim = 2
        else:
            raise ValueError("cannot auto detect, please specify feature_dim")

    if feature_dim < 0:
        feature_dim += tensor.dim()

    if not 0 <= feature_dim < tensor.dim():
        raise ValueError(f"feature_dim {feature_dim} out of range: [0, {tensor.dim()})")

    last_dim = tensor.dim() - 1
    if feature_dim == last_dim:
        return tensor

    dims = []
    for i in range(tensor.dim()):
        if i == feature_dim:
            dims.append(last_dim)
        elif i < feature_dim:
            dims.append(i)
        else:  # i > feature_dim
            dims.append(i - 1)

    return tensor.permute(*dims)


def _get_move_to_last_permutes(ndims, feature_dim):
    if feature_dim < 0:
        feature_dim += ndims
    if not 0 <= feature_dim < ndims:
        raise ValueError(f"feature_dim {feature_dim} out of range: [0, {ndims})")
    _move_to_last = [i for i in range(ndims) if i != feature_dim] + [feature_dim]
    return _move_to_last


def _get_move_from_last_permutes(ndims, feature_dim):
    if feature_dim < 0:
        feature_dim += ndims
    if not 0 <= feature_dim < ndims:
        raise ValueError(f"feature_dim {feature_dim} out of range: [0, {ndims})")
    _move_from_last = list(range(feature_dim)) + [ndims - 1] + [x - 1 for x in range(feature_dim + 1, ndims)]
    return _move_from_last


def get_permute_dims(ndims, feature_dim):
    if feature_dim < 0:
        feature_dim += ndims
    if not 0 <= feature_dim < ndims:
        raise ValueError(f"feature_dim {feature_dim} out of range: [0, {ndims})")
    _move_to_last = [i for i in range(ndims) if i != feature_dim] + [feature_dim]
    _move_from_last = list(range(feature_dim)) + [ndims - 1] + [x - 1 for x in range(feature_dim + 1, ndims)]
    return _move_to_last, _move_from_last


class HyperConnectionLayer(torch.nn.Module):
    """
    Implementation of `Hyper-Connection` ICLR2025.
    Paper: https://arxiv.org/pdf/2409.19606
    Author's blog: https://zhuanlan.zhihu.com/p/20810468231
    """

    def __init__(
        self,
        module: torch.nn.Module,
        expansion,
        idx,
        dynamic=False,
        d=None,  # dimension of the tensor, required if dynamic=True
        feature_dim=None,  # default treated as -1
        close=False,
        agg=None,  # used only when close=True
    ):
        super().__init__()
        self.operation = module
        self.n = expansion
        self.idx = idx
        self.dynamic = dynamic
        self.d = d
        self.feature_dim = None if feature_dim == -1 else feature_dim
        self.close = close
        self.agg = agg

        ar = torch.eye(self.n)
        am = torch.eye(self.n)[self.idx % self.n].unsqueeze(1)
        self.static_alpha = torch.nn.Parameter(torch.cat([am, ar], dim=1))  # (n,n+1)
        self.static_beta = torch.nn.Parameter(torch.ones(self.n))  # (n,)
        if self.dynamic:
            assert d is not None, "d should not be None in dynamic hyperconnection"
            self.dynamic_alpha_fn = torch.nn.Parameter(torch.zeros((d, self.n + 1)))
            self.dynamic_alpha_scale = torch.nn.Parameter(torch.ones(1) * 0.01)
            self.dynamic_beta_fn = torch.nn.Parameter(torch.zeros((d,)))
            self.dynamic_beta_scale = torch.nn.Parameter(torch.ones(1) * 0.01)
            self.layer_norm = torch.nn.LayerNorm(d)

        # auto fit feature dim
        self.ndims = None
        self._move_to_last = None
        self._move_from_last = None
        self.init_feature_dim_permute()

    def forward(self, hn, *args, **kwargs):
        """
        Perform the forward pass of the module.

        Assumptions:
        - Input has been expanded (unless in closed layer)
        - Module `self.operation` takes `hn` as the first input

        We use `nd` instead of `dn` for memory consistency.

        Args:
            hn: (..., n, d)

        Returns:
            refined_hn: (..., n, d)
                optionally with additional outputs if operation returns multiple values
        """

        if self.close:
            if self.feature_dim is not None:
                hn = self.move_feature_to_last(hn)
            hn = HyperConnectFactory._expand(hn, self.n)

        *ldims, _, _ = hn.shape

        if self.dynamic:
            hn_norm = self.layer_norm(hn)  # (...,nd)
            wc_weight = hn_norm @ self.dynamic_alpha_fn  # (...nd)(d,n+1)->(...n,n+1)
            wc_weight = F.tanh(wc_weight)  # (...n,n+1)
            dynamic_alpha = wc_weight * self.dynamic_alpha_scale  # (...n,n+1)
            alpha = self.static_alpha + dynamic_alpha  # (..., n,n+1)

            dc_weight = hn_norm @ self.dynamic_beta_fn  # (...,nd)(d,)->(..., n,)
            dc_weight = F.tanh(dc_weight)  # (...n,)
            dynamic_beta = dc_weight * self.dynamic_beta_scale  # (...n,)
            beta = self.static_beta + dynamic_beta  # (...n,)
        else:
            alpha = self.static_alpha  # (n,n+1)
            beta = self.static_beta  # (n,)

        # wide
        mix_h = alpha.transpose(-1, -2) @ hn  # (..., n+1, d)
        h_0 = mix_h[..., 0, :]  # (...,d)
        hn_ = mix_h[..., 1:, :]  # (...,nd)

        # deep
        if self.feature_dim is not None:
            h_0 = self.move_last_to_feature(h_0)
        _outs = self.operation(h_0, *args, **kwargs)  # (..., d)
        h0_o, *others = _outs if isinstance(_outs, tuple) else (_outs,)
        if self.feature_dim is not None:
            h0_o = self.move_feature_to_last(h0_o)
        h0_b = torch.einsum("...d, ...n->...nd", h0_o, beta)  # (..., nd)
        h_out = hn_ + h0_b  # (...,nd)

        # out
        if self.close:
            h_out = HyperConnectFactory._mix(h_out, agg=self.agg)
            if self.feature_dim is not None:
                self.move_last_to_feature(h_out)
        return (h_out, *others) if others else h_out

    def move_feature_to_last(self, tensor: torch.Tensor):
        if self.feature_dim == -1:
            return tensor
        if self.ndims is None:
            self.ndims = tensor.dim()
            self.init_feature_dim_permute()
        return tensor.permute(*self._move_to_last)

    def move_last_to_feature(self, tensor):
        if self.feature_dim == -1:
            return tensor
        if self.ndims is None:
            self.ndims = tensor.dim()
            self.init_feature_dim_permute()
        return tensor.permute(*self._move_from_last)

    def init_feature_dim_permute(self):
        if (self.feature_dim is None) or (self.ndims is None):
            return
        self._move_to_last, self._move_from_last = get_permute_dims(self.ndims, self.feature_dim)


class HyperConnectFactory:

    def __init__(self, expansion=2, dynamic=False, d=None, agg=None, feature_dim=None):
        self.n = expansion
        self.dynamic = dynamic
        self.d = d
        self.agg = agg
        self.feature_dim = feature_dim

        # hid
        self._global_indx = 0

    def layer(self, module: torch.nn.Module, dynamic=None, d=None, feature_dim=None) -> HyperConnectionLayer:
        dynamic = dynamic or self.dynamic
        d = d or self.d
        feature_dim = feature_dim or self.feature_dim

        idx = self._global_indx
        hc_ayer = HyperConnectionLayer(
            module,
            expansion=self.n,
            idx=idx,
            dynamic=dynamic,
            d=d,
            feature_dim=feature_dim,
        )
        self._global_indx = idx + 1
        return hc_ayer

    def closedLayer(
        self, module: torch.nn.Module, dynamic=None, d=None, feature_dim=None, agg=None
    ) -> HyperConnectionLayer:
        dynamic = dynamic or self.dynamic
        d = d or self.d
        feature_dim = feature_dim or self.feature_dim
        agg = agg or self.agg
        assert agg is not None, "aggregation should not be None in closed layer"

        idx = self._global_indx
        hc_layer = HyperConnectionLayer(
            module,
            expansion=self.n,
            idx=idx,
            dynamic=dynamic,
            d=d,
            feature_dim=feature_dim,
            close=True,
            agg=agg,
        )
        self._global_indx = idx + 1
        return hc_layer

    def Layer(self, *args, **kwargs):
        """alias for `layer`"""
        return self.layer(*args, **kwargs)

    def expand(self, h):
        return HyperConnectFactory._expand(h, self.n)

    def mix(self, hn, dim=-2, agg=None):
        if agg is None:
            agg = self.agg
        return HyperConnectFactory._mix(hn, dim, agg)

    def move_feature_to_last(self, tensor: torch.Tensor, feature_dim=None):
        feature_dim = feature_dim or self.feature_dim
        return HyperConnectFactory._move_feature_to_last(tensor, feature_dim)

    def move_last_to_feature(self, tensor: torch.Tensor, feature_dim=None):
        feature_dim = feature_dim or self.feature_dim
        return HyperConnectFactory._move_last_to_feature(tensor, feature_dim)

    @staticmethod
    def _expand(h, n):
        *_dims, d = h.shape
        return h.unsqueeze(dim=-2).expand(*_dims, n, d)

    @staticmethod
    def _mix(hn, dim=-2, agg="sum"):
        assert agg is not None, "aggregation should not be None"
        if agg == "sum":
            return hn.sum(dim=dim)
        elif agg == "mean":
            return hn.mean(dim=dim)
        else:
            raise TypeError(f"Unknown agg: {agg}")

    @staticmethod
    def _move_feature_to_last(tensor: torch.Tensor, feature_dim: int):
        assert feature_dim is not None
        _move_to_last = _get_move_to_last_permutes(tensor.dim(), feature_dim)
        return tensor.permute(*_move_to_last)

    @staticmethod
    def _move_last_to_feature(tensor: torch.Tensor, feature_dim: int):
        assert feature_dim is not None
        _move_from_last = _get_move_from_last_permutes(tensor.dim(), feature_dim)
        return tensor.permute(*_move_from_last)
