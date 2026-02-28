from qqtools.torch.nn import utils as nn_utils


def test_nn_utils_import_has_torch_symbol():
    assert hasattr(nn_utils, "torch")
