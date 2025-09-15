import torch

from .nonlinear import get_nonlinear


def build_qmlp_layers(
    fc_neurons,
    norm=None,
    activation="silu",
    bias=True,
    last_bias=True,
    last_norm=False,
    last_activation=False,
):
    modules = []
    assert len(fc_neurons) > 1
    for i in range(1, len(fc_neurons)):
        in_dim = fc_neurons[i - 1]
        to_dim = fc_neurons[i]
        is_last = i == len(fc_neurons) - 1

        if i == len(fc_neurons) - 1 and (not last_bias):
            use_bias = False
        else:
            use_bias = bias
        modules.append(torch.nn.Linear(in_dim, to_dim, bias=use_bias))

        if not is_last or last_norm:
            if norm == "ln":
                modules.append(torch.nn.LayerNorm(to_dim))
            elif norm is None or norm == "":
                pass
            else:
                raise ValueError(f"Invalid norm type: {norm}")

        if not is_last or last_activation:
            if activation is not None and activation != "":
                if isinstance(activation, str):
                    activation = get_nonlinear(activation)
                else:
                    assert callable(activation), f"activation {activation} is not callable"
                modules.append(activation)

    return torch.nn.Sequential(*modules)


class qMLP(torch.nn.Module):
    """
    easy MLP
    """

    def __init__(
        self,
        fc_neurons,
        norm=None,
        activation="silu",
        bias=True,
        last_bias=True,
        last_norm=False,
        last_activation=False,
        weight_init=torch.nn.init.xavier_uniform_,
        bias_init=torch.nn.init.zeros_,
    ):
        super().__init__()
        self.layers = build_qmlp_layers(
            fc_neurons,
            norm,
            activation,
            bias,
            last_bias,
            last_norm,
            last_activation,
        )

        # cover the default behavior of nn.Linear
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.reset_parameters()

    def forward(self, inpts):
        o = self.layers(inpts)
        return o

    def reset_parameters(self):
        for module in self.layers:
            if isinstance(module, torch.nn.Linear):
                self.weight_init(module.weight)
                if module.bias is not None:
                    self.bias_init(module.bias)
