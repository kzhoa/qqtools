"""Functional tests for no-weight-decay parameter group auto-discovery."""

import warnings

import pytest
import torch
import torch.nn as nn

from qqtools.plugins.qpipeline.entry_utils.no_decay import (
    build_param_groups,
    collect_no_decay_params,
)


# ---------------------------------------------------------------------------
# Test Models
# ---------------------------------------------------------------------------


class PlainModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x):
        return self.fc(x)


class NormWithNoDecay(nn.LayerNorm):
    def no_decay(self):
        return ["weight", "bias"]


class ModelWithNoDecayNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.norm = NormWithNoDecay(4)

    def forward(self, x):
        return self.norm(self.fc(x))


class BlockWithPosEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, 4, 8))
        self.norm = NormWithNoDecay(8)
        self.linear = nn.Linear(8, 8)

    def no_decay(self):
        return ["pos_embed"]

    def forward(self, x):
        return self.linear(self.norm(x + self.pos_embed))


class NestedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = BlockWithPosEmbed()
        self.head = nn.Linear(8, 2)

    def forward(self, x):
        return self.head(self.encoder(x))


class DeepDeclareModel(nn.Module):
    """Uses no_decay_deep to stop recursion."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 4)
        self.norm = NormWithNoDecay(4)

    def no_decay_deep(self):
        return ["encoder.bias", "norm.weight"]

    def forward(self, x):
        return self.norm(self.encoder(x))


class BothMethodsModel(nn.Module):
    """Has both no_decay and no_decay_deep — deep wins."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def no_decay(self):
        return ["fc.weight"]

    def no_decay_deep(self):
        return ["fc.bias"]

    def forward(self, x):
        return self.fc(x)


class ChildWithNoDecay(nn.Module):
    def __init__(self):
        super().__init__()
        self.param_a = nn.Parameter(torch.randn(4))

    def no_decay(self):
        return ["param_a"]

    def forward(self, x):
        return x + self.param_a


class ParentDeepStopsChild(nn.Module):
    """Parent uses no_decay_deep — child's no_decay should NOT be visited."""

    def __init__(self):
        super().__init__()
        self.child = ChildWithNoDecay()
        self.fc = nn.Linear(4, 4)

    def no_decay_deep(self):
        return ["fc.bias"]

    def forward(self, x):
        return self.fc(self.child(x))


class ModelWithNoneChild(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self._modules["empty_slot"] = None

    def forward(self, x):
        return self.fc(x)


class ModelWithBogusName(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def no_decay(self):
        return ["nonexistent_param"]

    def forward(self, x):
        return self.fc(x)


class ModelWithFrozenNoDecay(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.frozen_param = nn.Parameter(torch.randn(4), requires_grad=False)

    def no_decay(self):
        return ["frozen_param", "fc.bias"]

    def forward(self, x):
        return self.fc(x) + self.frozen_param


# ---------------------------------------------------------------------------
# TestCollectNoDecayParams
# ---------------------------------------------------------------------------


class TestCollectNoDecayParams:
    def test_plain_model_returns_empty(self):
        model = PlainModel()
        result = collect_no_decay_params(model)
        assert result == set()

    def test_no_decay_collects_local_names(self):
        model = ModelWithNoDecayNorm()
        result = collect_no_decay_params(model)
        assert result == {"norm.weight", "norm.bias"}

    def test_no_decay_continues_recursion(self):
        model = NestedModel()
        result = collect_no_decay_params(model)
        assert "encoder.pos_embed" in result
        assert "encoder.norm.weight" in result
        assert "encoder.norm.bias" in result

    def test_no_decay_deep_stops_recursion(self):
        model = ParentDeepStopsChild()
        result = collect_no_decay_params(model)
        # Parent declares only fc.bias via no_decay_deep
        assert result == {"fc.bias"}
        # Child's no_decay() for param_a is NOT visited
        assert "child.param_a" not in result

    def test_both_methods_deep_wins(self):
        model = BothMethodsModel()
        result = collect_no_decay_params(model)
        assert result == {"fc.bias"}
        assert "fc.weight" not in result

    def test_nested_fully_qualified_names(self):
        model = NestedModel()
        result = collect_no_decay_params(model)
        expected = {"encoder.pos_embed", "encoder.norm.weight", "encoder.norm.bias"}
        assert expected == result

    def test_none_child_no_crash(self):
        model = ModelWithNoneChild()
        result = collect_no_decay_params(model)
        assert result == set()

    def test_deduplication(self):
        """Duplicate names from overlapping declarations are deduplicated."""

        class DuplicatingNorm(nn.LayerNorm):
            def no_decay(self):
                return ["weight", "weight"]  # duplicate in return list

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = DuplicatingNorm(4)

            def forward(self, x):
                return self.norm(x)

        model = M()
        result = collect_no_decay_params(model)
        assert result == {"norm.weight"}


# ---------------------------------------------------------------------------
# TestParamGroupConstruction
# ---------------------------------------------------------------------------


class TestParamGroupConstruction:
    def test_zero_weight_decay_single_group(self):
        model = ModelWithNoDecayNorm()
        params = {"lr": 0.001, "weight_decay": 0.0, "betas": [0.9, 0.999]}
        groups = build_param_groups(model, params)
        assert len(groups) == 1
        assert groups[0]["weight_decay"] == 0.0

    def test_splits_into_two_groups(self):
        model = ModelWithNoDecayNorm()
        params = {"lr": 0.001, "weight_decay": 0.01, "betas": [0.9, 0.999]}
        groups = build_param_groups(model, params)
        assert len(groups) == 2
        assert groups[0]["weight_decay"] == 0.01
        assert groups[1]["weight_decay"] == 0.0

    def test_decay_group_has_correct_params(self):
        model = ModelWithNoDecayNorm()
        params = {"lr": 0.001, "weight_decay": 0.01}
        groups = build_param_groups(model, params)
        # fc.weight + fc.bias = 2 tensors in decay group
        decay_param_count = len(groups[0]["params"])
        no_decay_param_count = len(groups[1]["params"])
        assert decay_param_count == 2  # fc.weight, fc.bias
        assert no_decay_param_count == 2  # norm.weight, norm.bias

    def test_frozen_params_excluded(self):
        model = ModelWithFrozenNoDecay()
        params = {"lr": 0.001, "weight_decay": 0.01}
        groups = build_param_groups(model, params)
        total_in_groups = sum(len(g["params"]) for g in groups)
        trainable_count = sum(1 for p in model.parameters() if p.requires_grad)
        assert total_in_groups == trainable_count

    def test_nonexistent_name_warns(self):
        model = ModelWithBogusName()
        params = {"lr": 0.001, "weight_decay": 0.01}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            groups = build_param_groups(model, params)
            assert len(w) == 1
            assert "nonexistent_param" in str(w[0].message)
        # Single group since no valid no-decay params
        assert len(groups) == 1

    def test_all_hyperparams_shared(self):
        model = ModelWithNoDecayNorm()
        params = {"lr": 0.001, "weight_decay": 0.01, "betas": [0.9, 0.999], "eps": 1e-8}
        groups = build_param_groups(model, params)
        assert groups[0]["lr"] == 0.001
        assert groups[0]["betas"] == [0.9, 0.999]
        assert groups[0]["eps"] == 1e-8
        assert groups[1]["lr"] == 0.001
        assert groups[1]["betas"] == [0.9, 0.999]
        assert groups[1]["eps"] == 1e-8
        assert "weight_decay" not in groups[1] or groups[1]["weight_decay"] == 0.0

    def test_empty_no_decay_single_group(self):
        model = PlainModel()
        params = {"lr": 0.001, "weight_decay": 0.01}
        groups = build_param_groups(model, params)
        assert len(groups) == 1

    def test_no_trainable_params(self):
        model = PlainModel()
        for p in model.parameters():
            p.requires_grad = False
        params = {"lr": 0.001, "weight_decay": 0.01}
        groups = build_param_groups(model, params)
        assert len(groups) == 1
        assert len(groups[0]["params"]) == 0


# ---------------------------------------------------------------------------
# TestPrepareOptimizerIntegration
# ---------------------------------------------------------------------------


class TestPrepareOptimizerIntegration:
    def _make_args(self, weight_decay=0.01):
        import qqtools as qt

        return qt.qDict(
            {
                "optim": {
                    "optimizer": "AdamW",
                    "optimizer_params": {
                        "lr": 0.001,
                        "weight_decay": weight_decay,
                        "betas": [0.9, 0.999],
                        "eps": 1e-8,
                    },
                },
            }
        )

    def test_plain_model_single_group(self, monkeypatch):
        import qqtools as qt
        from qqtools.plugins.qpipeline.entry_utils.optimizer import prepare_optimizer

        monkeypatch.setattr(qt.qdist, "get_rank", lambda: 0)
        args = self._make_args(weight_decay=0.01)
        model = PlainModel()
        optimizer = prepare_optimizer(args, model)
        assert len(optimizer.param_groups) == 1

    def test_model_with_no_decay_two_groups(self, monkeypatch):
        import qqtools as qt
        from qqtools.plugins.qpipeline.entry_utils.optimizer import prepare_optimizer

        monkeypatch.setattr(qt.qdist, "get_rank", lambda: 0)
        args = self._make_args(weight_decay=0.01)
        model = ModelWithNoDecayNorm()
        optimizer = prepare_optimizer(args, model)
        assert len(optimizer.param_groups) == 2
        assert optimizer.param_groups[0]["weight_decay"] == 0.01
        assert optimizer.param_groups[1]["weight_decay"] == 0.0

    def test_zero_weight_decay_single_group(self, monkeypatch):
        import qqtools as qt
        from qqtools.plugins.qpipeline.entry_utils.optimizer import prepare_optimizer

        monkeypatch.setattr(qt.qdist, "get_rank", lambda: 0)
        args = self._make_args(weight_decay=0.0)
        model = ModelWithNoDecayNorm()
        optimizer = prepare_optimizer(args, model)
        assert len(optimizer.param_groups) == 1

    def test_rank_zero_logging(self, monkeypatch, capsys):
        import qqtools as qt
        from qqtools.plugins.qpipeline.entry_utils.optimizer import prepare_optimizer

        monkeypatch.setattr(qt.qdist, "get_rank", lambda: 0)
        args = self._make_args(weight_decay=0.01)
        model = ModelWithNoDecayNorm()
        prepare_optimizer(args, model)
        captured = capsys.readouterr()
        assert "no-weight-decay group" in captured.out

    def test_non_rank_zero_no_logging(self, monkeypatch, capsys):
        import qqtools as qt
        from qqtools.plugins.qpipeline.entry_utils.optimizer import prepare_optimizer

        monkeypatch.setattr(qt.qdist, "get_rank", lambda: 1)
        args = self._make_args(weight_decay=0.01)
        model = ModelWithNoDecayNorm()
        prepare_optimizer(args, model)
        captured = capsys.readouterr()
        assert "no-weight-decay" not in captured.out

    def test_optimizer_step_works(self, monkeypatch):
        """Verify optimizer can actually perform a step with param groups."""
        import qqtools as qt
        from qqtools.plugins.qpipeline.entry_utils.optimizer import prepare_optimizer

        monkeypatch.setattr(qt.qdist, "get_rank", lambda: 0)
        args = self._make_args(weight_decay=0.01)
        model = ModelWithNoDecayNorm()
        optimizer = prepare_optimizer(args, model)

        x = torch.randn(2, 4)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
