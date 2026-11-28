"""Fresh-process import isolation and backend binding for qneighbor."""

import importlib.util
import subprocess
import sys

import pytest
import torch

pytestmark = pytest.mark.integration


def test_neutral_import_is_lazy_with_both_backends_absent():
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
class Block:
    def find_spec(self, name, *args):
        if name.split('.')[0] in ('warp', 'triton'):
            raise ModuleNotFoundError(name, name=name)
sys.meta_path.insert(0, Block())
from qqtools.plugins import qneighbor
from qqtools.plugins.qneighbor import cuda
assert 'warp' not in sys.modules and 'triton' not in sys.modules
try:
    cuda.neighbor_list_obc
except ImportError as exc:
    assert 'qqtools[qneighbor]' in str(exc) and 'qqtools[qneighbor-triton]' in str(exc)
else:
    raise AssertionError('Missing backends accepted')
""",
        ],
        check=True,
        timeout=30,
    )


def test_cpu_does_not_require_optional_backends():
    script = """
import sys
class Block:
    def find_spec(self, name, *args):
        if name.split(".")[0] in ("warp", "triton", "ase"):
            raise ModuleNotFoundError(name, name=name)
sys.meta_path.insert(0, Block())
import torch
from qqtools.plugins.qneighbor.cpu import neighbor_list
assert "qqtools.plugins.qneighbor.cuda" not in sys.modules
edges, offsets = neighbor_list(torch.zeros((2, 3)), 1.)
assert edges.shape == (2, 2) and offsets.shape == (2, 3) and not bool(offsets.any())
assert neighbor_list(torch.zeros((1, 3)), 1.1, cell=torch.eye(3), pbc=True)[0].shape == (2, 6)
"""
    subprocess.run([sys.executable, "-c", script], check=True, timeout=30)


def test_cuda_module_import_does_not_load_optional_backends():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
class Block:
    def find_spec(self, name, *args):
        if name.split('.')[0] in ('warp', 'triton'):
            raise AssertionError('Unexpected backend import: ' + name)
sys.meta_path.insert(0, Block())
from qqtools.plugins.qneighbor import cuda
assert cuda._selected is None
assert 'warp' not in sys.modules and 'triton' not in sys.modules
try:
    cuda.unknown_neighbor_function
except AttributeError:
    pass
else:
    raise AssertionError('Unknown export accepted')
""",
        ],
        check=True,
        timeout=30,
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("selected", ["warp", "triton"])
def test_neutral_binding_executes_with_other_backend_absent(selected):
    import subprocess
    import sys

    if importlib.util.find_spec(selected) is None:
        pytest.skip(f"{selected} extra required")
    script = """
import importlib.util
import sys
blocked = 'triton' if sys.argv[1] == 'warp' else 'warp'
class Block:
    def find_spec(self, name, *args):
        if name.split('.')[0] == blocked:
            raise ModuleNotFoundError(name, name=name)
sys.meta_path.insert(0, Block())
original = importlib.util.find_spec
def find_spec(name, package=None):
    if name.split('.')[0] == blocked:
        return None
    return original(name, package)
importlib.util.find_spec = find_spec
from qqtools.plugins import qneighbor
from qqtools.plugins.qneighbor import cuda
assert 'warp' not in sys.modules and 'triton' not in sys.modules
# Explicit namespaces must work while the automatic resolver is disabled.
from importlib import import_module
explicit = import_module('qqtools.plugins.qneighbor.cuda.' + sys.argv[1])
assert cuda._selected is None
assert explicit.radius_graph is explicit.neighbor_list_obc
saved = cuda._ensure_backend
def forbidden(*args, **kwargs):
    raise AssertionError('Explicit import used default backend selection')
cuda._ensure_backend = forbidden
import torch
p = torch.tensor([[0.,0,0],[.5,0,0]], device='cuda')
b = torch.zeros(2, device='cuda', dtype=torch.long)
c = torch.eye(3, device='cuda') * 4
assert explicit.radius_graph(p,1.,b).tolist() == [[1,0],[0,1]]
e, s = explicit.neighbor_list(p,1.,b)
assert e.tolist() == [[1,0],[0,1]] and s.shape == (2,3) and not bool(s.any())
e, s = explicit.neighbor_list_pbc(p,1.,cell=c)
assert e.tolist() == [[1,0],[0,1]] and not bool(s.any())
e, s = explicit.neighbor_list(p,1.,b,cell=c,pbc=True)
assert e.tolist() == [[1,0],[0,1]] and not bool(s.any())
assert cuda._selected is None
cuda._ensure_backend = saved
search = qneighbor.neighbor_list
backend = cuda._selected
assert backend is explicit
# Importing the public CUDA API must work with just the selected backend installed.
from qqtools.plugins.qneighbor.cuda import *
assert neighbor_list is search is explicit.neighbor_list
assert neighbor_list_obc is explicit.neighbor_list_obc
assert neighbor_list_pbc is explicit.neighbor_list_pbc
assert radius_graph is explicit.radius_graph
assert cuda.radius_graph is backend.radius_graph
assert qneighbor.radius_graph is backend.radius_graph
assert qneighbor.neighbor_list_obc is qneighbor.radius_graph
assert qneighbor.neighbor_list_pbc is backend.neighbor_list_pbc
search = qneighbor.neighbor_list
# Subsequent calls must not access initialization or dependency-selection code.
def fail(*args, **kwargs):
    raise AssertionError('Backend selection repeated')
cuda._ensure_backend = cuda._load_backend = fail
import torch
pos = torch.tensor([[0.,0,0],[.5,0,0]], device='cuda')
ids = torch.zeros(2,device='cuda',dtype=torch.long)
cell = torch.eye(3,device='cuda') * 4
for _ in range(2):
    edges = qneighbor.radius_graph(positions=pos,cutoff=1.,batch=ids,max_neighbors=32)
    assert edges.tolist() == [[1,0],[0,1]]
    generic, generic_shift = search(pos,1.,ids)
    assert generic.tolist() == edges.tolist() and generic_shift.shape == (2,3)
    assert not bool(generic_shift.any())
    generic, generic_shift = search(pos,1.,ids,cell=cell,pbc=True)
    assert generic.tolist() == edges.tolist() and not bool(generic_shift.any())
    public_edges, public_shift = qneighbor.neighbor_list_pbc(pos,1.,cell=cell)
    assert public_edges.tolist() == edges.tolist() and not bool(public_shift.any())
    edges,shift = qneighbor.neighbor_list_pbc(pos,1.,ids,32,cell=cell,pbc=True)
    assert edges.tolist() == [[1,0],[0,1]] and not bool(shift.any())
    edges,shift = explicit.neighbor_list_pbc(pos,1.,ids,32,cell=cell)
    assert edges.tolist() == [[1,0],[0,1]] and not bool(shift.any())
assert not any(name.split('.')[0] == blocked for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", script, selected], check=True, timeout=120)


def test_warp_is_loaded_only_during_search():
    script = """
import sys

class BlockWarp:
    def find_spec(self, name, *args):
        if name == "warp":
            raise ModuleNotFoundError(name, name=name)

sys.meta_path.insert(0, BlockWarp())
import qqtools
from qqtools.plugins import qneighbor
from qqtools.plugins.qneighbor.cuda.warp import neighbor_list_pbc
import torch

assert "warp" not in sys.modules
assert qqtools.qDict({"answer": 1}).answer == 1
pos = torch.zeros((1, 3))
batch = torch.zeros(1, dtype=torch.long)
pbc = torch.zeros(3, dtype=torch.bool)
try:
    neighbor_list_pbc(pos,1.,batch,1,pbc=pbc)
except ImportError as exc:
    assert "qqtools[qneighbor]" in str(exc)
else:
    raise AssertionError("Missing Warp dependency was accepted")
"""
    subprocess.run([sys.executable, "-c", script], check=True, timeout=30)
