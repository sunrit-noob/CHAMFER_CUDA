import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

__version__ = "0.1"

def load_cpp_ext(ext_name):
    root_dir = os.path.join(os.path.split(__file__)[0])
    ext_csrc = os.path.join(root_dir, "src")
    ext_path = os.path.join(ext_csrc, "_ext", ext_name)
    os.makedirs(ext_path, exist_ok=True)
    assert torch.cuda.is_available(), "torch.cuda.is_available() is False."
    ext_sources = [
        os.path.join(ext_csrc, "{}.cpp".format(ext_name)),
        os.path.join(ext_csrc, "{}.cu".format(ext_name))
    ]
    extra_cuda_cflags = [
        "-DCUDA_HAS_FP16=1",
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
    ]
    ext = load(
        name=ext_name,
        sources=ext_sources,
        extra_cflags=["-O2"],
        build_directory=ext_path,
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=False,
        with_cuda=True
    )
    return ext


_cd = load_cpp_ext("chamfer")


def chamfer(pc1, pc2):
    d = _cd.chamfer_distance(pc1, pc2)
    return d


def _T(t, mode=False):
    if mode:
        return t.transpose(0, 1).contiguous()
    else:
        return t


class CD(nn.Module):

    def __init__(self, transpose_mode=False):
        super(CD, self).__init__()
        self._t = transpose_mode

    def forward(self, pc1, pc2):
        assert pc1.size(0) == pc2.size(0), "pc1.shape={} != pc2.shape={}".format(pc1.shape, pc2.shape)
        with torch.no_grad():
            batch_size = pc1.size(0)
            D = []
            for bi in range(batch_size):
                r, q = _T(pc1[bi], self._t), _T(pc2[bi], self._t)
                d = chamfer(r.float(), q.float())
                D.append(d)
            D = torch.stack(D, dim=0)
        return D.squeeze()
