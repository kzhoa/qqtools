from collections import defaultdict

import numpy as np
import pytest
import torch

import qqtools as qt
from qqtools.torch.qdataset import collate_graph_samples, determine_graph_key_types, smart_combine

# --- Pytest Test Cases ---


# Helper function to create dummy graph data
def create_sample(
    num_nodes, edge_index, node_features=None, graph_features=None, edge_features=None, num_nodes_explicit=None
):
    sample = {}
    if edge_index is not None:
        sample["edge_index"] = torch.tensor(edge_index, dtype=torch.long)

    if num_nodes_explicit is not None:
        sample["num_nodes"] = num_nodes_explicit

    if node_features is not None:
        sample["x"] = torch.tensor(node_features, dtype=torch.float)
    if graph_features is not None:
        sample["graph_attr"] = torch.tensor(graph_features, dtype=torch.float)
    if edge_features is not None:
        sample["edge_attr"] = torch.tensor(edge_features, dtype=torch.float)

    # Add a dummy node attribute if only graph attr exists to test node attr inference
    if node_features is None and num_nodes is not None and len(sample) > 0 and "x" not in sample:
        sample["dummy_node_attr"] = torch.randn(num_nodes, 1)

    # Add a dummy edge attribute if only graph attr exists to test edge attr inference
    if edge_features is None and edge_index is not None and len(edge_index[0]) > 0 and "edge_attr" not in sample:
        sample["dummy_edge_attr"] = torch.randn(len(edge_index[0]), 1)

    # Ensure num_nodes is present if node features are absent but num_nodes is known
    if num_nodes is not None and "num_nodes" not in sample:
        sample["num_nodes"] = num_nodes

    return sample


# --- Test Suite ---
# Section: Anchored Paths
# Validate common collation paths with num_nodes and/or edge_index, including basic edge cases.


def test_empty_batch():
    """Validate collation for an empty batch."""
    batch_list = []
    result = collate_graph_samples(batch_list)
    assert result == {}


def test_single_graph_no_edge_index():
    """Validate collation for a single graph without edge_index but with num_nodes."""
    nodes = 3
    sample = create_sample(num_nodes=nodes, edge_index=None, node_features=[[1], [2], [3]], graph_features=[10])
    batch_list = [sample]
    result = collate_graph_samples(batch_list)

    assert "batch" in result
    assert torch.equal(result["batch"], torch.tensor([0, 0, 0]))
    assert "x" in result
    assert torch.equal(result["x"], torch.tensor([[1], [2], [3]]))
    assert "graph_attr" in result
    assert torch.equal(result["graph_attr"], torch.tensor([10]))
    assert "num_nodes" not in result  # num_nodes is usually removed after collation


def test_single_graph_with_edge_index():
    """Validate collation for a single graph with edge_index."""
    edge_index = [[0, 1, 1], [1, 0, 2]]  # 3 edges, max node index 2 -> 3 nodes
    sample = create_sample(
        num_nodes=3,
        edge_index=edge_index,
        node_features=[[1], [2], [3]],
        edge_features=[[10], [11], [12]],
        graph_features=[100],
    )
    batch_list = [sample]
    result = collate_graph_samples(batch_list)

    assert "batch" in result
    assert torch.equal(result["batch"], torch.tensor([0, 0, 0]))
    assert "x" in result
    assert torch.equal(result["x"], torch.tensor([[1], [2], [3]]))
    assert "edge_attr" in result
    assert torch.equal(result["edge_attr"], torch.tensor([[10], [11], [12]]))
    assert "graph_attr" in result
    assert torch.equal(result["graph_attr"], torch.tensor([100]))
    assert "edge_index" in result
    assert torch.equal(result["edge_index"], torch.tensor(edge_index))
    assert "num_nodes" not in result


def test_two_graphs_different_nodes_no_edge_index():
    """Validate batching for two graphs with varying nodes and no edge_index."""
    sample1 = create_sample(num_nodes=3, edge_index=None, node_features=[[1], [2], [3]], graph_features=[10])
    sample2 = create_sample(num_nodes=2, edge_index=None, node_features=[[4], [5]], graph_features=[20])
    batch_list = [sample1, sample2]
    result = collate_graph_samples(batch_list)

    assert "batch" in result
    assert torch.equal(result["batch"], torch.tensor([0, 0, 0, 1, 1]))
    assert "x" in result
    assert torch.equal(result["x"], torch.tensor([[1], [2], [3], [4], [5]]))
    assert "graph_attr" in result
    assert torch.equal(result["graph_attr"], torch.tensor([10, 20]))
    assert "edge_index" not in result


def test_two_graphs_different_nodes_with_edge_index():
    """Validate batching for two graphs with varying nodes and edge_index."""
    sample1 = create_sample(
        num_nodes=3,
        edge_index=[[0, 1], [1, 2]],
        node_features=[[1], [2], [3]],
        edge_features=[[10], [11]],
        graph_features=[100],
    )
    sample2 = create_sample(
        num_nodes=2,
        edge_index=[[0, 1], [1, 0]],
        node_features=[[4], [5]],
        edge_features=[[20], [21]],
        graph_features=[200],
    )
    batch_list = [sample1, sample2]
    result = collate_graph_samples(batch_list)

    assert "batch" in result
    assert torch.equal(result["batch"], torch.tensor([0, 0, 0, 1, 1]))
    assert "x" in result
    assert torch.equal(result["x"], torch.tensor([[1], [2], [3], [4], [5]]))
    assert "edge_attr" in result
    assert torch.equal(result["edge_attr"], torch.tensor([[10], [11], [20], [21]]))
    assert "graph_attr" in result
    assert torch.equal(result["graph_attr"], torch.tensor([100, 200]))
    assert "edge_index" in result
    # Expected edge_index: first graph's [0,1] shifted by 0, second graph's [0,1] shifted by 3
    expected_edge_index = torch.tensor([[0, 1, 3, 4], [1, 2, 4, 3]])
    assert torch.equal(result["edge_index"], expected_edge_index)


def test_graphs_with_only_graph_attributes_and_num_nodes():
    """Validate graphs containing only graph attributes with explicit num_nodes."""
    sample1 = {"num_nodes": 2, "graph_attr": torch.tensor([10.0])}
    sample2 = {"num_nodes": 3, "graph_attr": torch.tensor([20.0])}
    batch_list = [sample1, sample2]

    # Manually specify key types. Node/edge sets are empty.
    key_types = {"graph": {"graph_attr"}, "node": set(), "edge": set()}
    result = collate_graph_samples(batch_list, key_types=key_types)

    assert "batch" in result
    assert torch.equal(result["batch"], torch.tensor([0, 0, 1, 1, 1]))

    assert "graph_attr" in result
    assert torch.equal(result["graph_attr"], torch.tensor([10.0, 20.0]))

    assert "edge_index" not in result

    assert "x" not in result
    assert "dummy_node_attr" not in result

    assert "num_nodes" not in result


def test_graphs_with_edge_index_but_no_node_attributes():
    """Validate node-count inference from edge_index when node attributes are absent."""
    sample1 = {"edge_index": torch.tensor([[0, 1], [1, 0]]), "graph_attr": torch.tensor([10.0])}  # Implies 2 nodes
    sample2 = {"edge_index": torch.tensor([[0], [0]]), "graph_attr": torch.tensor([20.0])}  # Implies 1 node
    batch_list = [sample1, sample2]

    result = collate_graph_samples(batch_list)

    assert "batch" in result
    # Graph 1 has 2 nodes, Graph 2 has 1 node. Total 3 nodes. Indices: [0, 0] for G1, [1] for G2.
    assert torch.equal(result["batch"], torch.tensor([0, 0, 1]))
    assert "graph_attr" in result
    assert torch.equal(result["graph_attr"], torch.tensor([10.0, 20.0]))
    assert "edge_index" in result
    # Expected edge_index: G1 edges [[0, 1], [1, 0]], G2 edges shifted [[2], [2]]
    # Combined: [[0, 1, 2], [1, 0, 2]]
    assert torch.equal(result["edge_index"], torch.tensor([[0, 1, 2], [1, 0, 2]]))
    assert "x" not in result
    assert "num_nodes" not in result


def test_determine_graph_key_types_marks_non_tensor_as_graph():
    """Validate non-tensor attributes are directly classified as graph attributes."""
    batch_list = [
        {
            "num_nodes": 2,
            "x": torch.randn(2, 3),
            "graph_note": "g0",
            "graph_meta": {"split": "train", "id": 1},
        },
        {
            "num_nodes": 3,
            "x": torch.randn(3, 3),
            "graph_note": "g1",
            "graph_meta": {"split": "val", "id": 2},
        },
    ]

    key_types = determine_graph_key_types(batch_list)
    assert key_types is not None
    assert "x" in key_types["node"]
    assert "graph_note" in key_types["graph"]
    assert "graph_meta" in key_types["graph"]


def test_collate_graph_samples_keeps_non_tensor_graph_attrs_as_list():
    """Validate str/dict graph attributes are aggregated as ordered lists."""
    batch_list = [
        {
            "num_nodes": 2,
            "x": torch.tensor([[1.0], [2.0]]),
            "graph_note": "first",
            "graph_meta": {"stage": "train", "epoch": 1},
        },
        {
            "num_nodes": 1,
            "x": torch.tensor([[3.0]]),
            "graph_note": "second",
            "graph_meta": {"stage": "val", "epoch": 2},
        },
    ]

    result = collate_graph_samples(batch_list)

    assert torch.equal(result["batch"], torch.tensor([0, 0, 1]))
    assert torch.equal(result["x"], torch.tensor([[1.0], [2.0], [3.0]]))
    assert result["graph_note"] == ["first", "second"]
    assert result["graph_meta"] == [
        {"stage": "train", "epoch": 1},
        {"stage": "val", "epoch": 2},
    ]


def test_graphs_with_ambiguous_shape_constant_but_different_meaning():
    """Validate node-attribute classification when shapes are constant across samples."""
    # Sample 1: 3 nodes, node feature dim 2
    sample1 = create_sample(
        num_nodes=3, edge_index=[[0, 1], [2, 0]], node_features=[[1, 1], [2, 2], [3, 3]], graph_features=[10]
    )
    # Sample 2: 3 nodes, node feature dim 2 (Same num_nodes and feature dim)
    sample2 = create_sample(
        num_nodes=3, edge_index=[[0, 1], [1, 2]], node_features=[[4, 4], [5, 5], [6, 6]], graph_features=[20]
    )
    batch_list = [sample1, sample2]

    # 'x' should be identified as node attribute because shape varies IF num_nodes varied.
    # However, since num_nodes is constant (3), determine_key_types might classify it as graph attr
    # if not careful. Let's ensure it's correctly classified as node attr.
    # The logic: `if not is_shape_constant:` OR `elif num_nodes0 != -1 and shape0 == num_nodes0:`
    # should correctly identify 'x' as node_attr here.
    result = collate_graph_samples(batch_list)

    assert "batch" in result
    assert torch.equal(result["batch"], torch.tensor([0, 0, 0, 1, 1, 1]))
    assert "x" in result
    # Concatenated node features
    assert torch.equal(result["x"], torch.tensor([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]]))
    assert "graph_attr" in result
    assert torch.equal(result["graph_attr"], torch.tensor([10, 20]))
    assert "edge_index" in result
    assert torch.equal(result["edge_index"], torch.tensor([[0, 1, 3, 4], [2, 0, 4, 5]]))


def test_edge_case_empty_edge_index():
    """Validate behavior when edge_index exists but is empty."""
    sample1 = {"edge_index": torch.empty((2, 0), dtype=torch.long), "num_nodes": 2, "x": torch.randn(2, 5)}
    sample2 = {"edge_index": torch.empty((2, 0), dtype=torch.long), "num_nodes": 3, "x": torch.randn(3, 5)}
    batch_list = [sample1, sample2]

    result = collate_graph_samples(batch_list)

    assert "batch" in result
    assert torch.equal(result["batch"], torch.tensor([0, 0, 1, 1, 1]))
    assert "x" in result
    assert result["x"].shape == (5, 5)  # Concatenated node features
    assert "edge_index" in result
    assert result["edge_index"].shape == (2, 0)  # Empty edge index tensor


def test_node_attribute_missing_in_one_sample():
    """Validate error behavior when a required node attribute is missing in one sample."""
    sample1 = create_sample(num_nodes=3, edge_index=[[0, 1], [1, 0]], node_features=[[1], [2], [3]])
    sample2 = {"edge_index": torch.tensor([[0, 1], [1, 0]]), "num_nodes": 2}  # Missing 'x'
    batch_list = [sample1, sample2]

    # This test expects the code to either raise an error due to inconsistent keys/attributes
    # or handle the missing attribute gracefully (e.g., by padding or skipping).
    # Given the current implementation's strict asserts, it's likely to fail.
    # We'll use pytest.raises to check for the expected assertion error.
    with pytest.raises(AssertionError, match="Not all keys are the same in the batch"):
        collate_graph_samples(batch_list)

    # If you wanted it to handle missing attributes, you'd need to modify the code
    # to use sample.get(k) more broadly and potentially add padding logic.


# Section: No-Anchor Paths
# Validate daily no-anchor workloads with fallback behavior and strict-mode failure expectations.


def test_determine_graph_key_types_default_fallback_marks_variable_keys_as_node_attr():
    """Validate default no-anchor inference maps variable keys to node attributes."""
    batch_list = [
        {
            "x": torch.randn(3, 4),
            "aux": torch.randn(3, 2),
            "graph_attr": torch.tensor([1.0]),
        },
        {
            "x": torch.randn(5, 4),
            "aux": torch.randn(5, 2),
            "graph_attr": torch.tensor([2.0]),
        },
    ]

    key_types = determine_graph_key_types(batch_list)
    assert key_types is not None
    assert key_types["node"] == {"x", "aux"}
    assert key_types["edge"] == set()
    assert key_types["graph"] == {"graph_attr"}


def test_collate_graph_samples_default_fallback_without_num_nodes_and_edge_index():
    """Validate default no-anchor collation for variable-length attributes."""
    batch_list = [
        {
            "x": torch.randn(3, 4),
            "aux": torch.randn(3, 2),
            "graph_attr": torch.tensor([1.0]),
        },
        {
            "x": torch.randn(5, 4),
            "aux": torch.randn(5, 2),
            "graph_attr": torch.tensor([2.0]),
        },
    ]

    result = collate_graph_samples(batch_list)

    assert "batch" in result
    assert torch.equal(result["batch"], torch.tensor([0, 0, 0, 1, 1, 1, 1, 1]))
    assert "x" in result and result["x"].shape == (8, 4)
    assert "aux" in result and result["aux"].shape == (8, 2)
    assert "graph_attr" in result and result["graph_attr"].shape == (2,)
    assert "edge_index" not in result


def test_no_anchor_default_mixed_variable_and_constant_attrs():
    """Validate mixed variable and constant attributes under default no-anchor flow."""
    batch_list = [
        {
            "x": torch.tensor([[1.0, 1.0], [2.0, 2.0]]),
            "aux": torch.tensor([[10.0], [20.0]]),
            "meta": torch.tensor([100.0]),
        },
        {
            "x": torch.tensor([[3.0, 3.0], [4.0, 4.0], [5.0, 5.0]]),
            "aux": torch.tensor([[30.0], [40.0], [50.0]]),
            "meta": torch.tensor([200.0]),
        },
    ]

    result = collate_graph_samples(batch_list)

    assert torch.equal(result["batch"], torch.tensor([0, 0, 1, 1, 1]))
    assert torch.equal(result["x"], torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0]]))
    assert torch.equal(result["aux"], torch.tensor([[10.0], [20.0], [30.0], [40.0], [50.0]]))
    assert torch.equal(result["meta"], torch.tensor([100.0, 200.0]))
    assert "edge_index" not in result


def test_no_anchor_default_all_constant_attrs_graph_only():
    """Validate graph-only output when all no-anchor attributes are constant-length."""
    batch_list = [
        {
            "graph_attr": torch.tensor([1.0]),
            "global_feat": torch.tensor([10.0, 11.0]),
        },
        {
            "graph_attr": torch.tensor([2.0]),
            "global_feat": torch.tensor([20.0, 21.0]),
        },
    ]

    result = collate_graph_samples(batch_list)

    assert result["batch"].numel() == 0
    assert result["batch"].dtype == torch.long
    assert torch.equal(result["graph_attr"], torch.tensor([1.0, 2.0]))
    assert torch.equal(result["global_feat"], torch.tensor([[10.0, 11.0], [20.0, 21.0]]))
    assert "edge_index" not in result


def test_graph_attr_normalize_compresses_consecutive_leading_singletons_arbitrary_dims():
    """Validate graph-level normalization compresses consecutive leading singleton dimensions."""
    batch_list = [
        {
            "num_nodes": 2,
            "x": torch.tensor([[1.0], [2.0]]),
            "g_vec": torch.tensor([1.0, 2.0, 3.0]),
            "g_row": torch.tensor([[4.0, 5.0, 6.0]]),
            "g_scalar_like": torch.tensor([7.0]),
            "g_scalar": torch.tensor(8.0),
            "g_high": torch.arange(6.0).reshape(1, 1, 2, 3),
            "g_no_compress": torch.arange(6.0).reshape(2, 1, 3),
        },
        {
            "num_nodes": 1,
            "x": torch.tensor([[3.0]]),
            "g_vec": torch.tensor([10.0, 20.0, 30.0]),
            "g_row": torch.tensor([[40.0, 50.0, 60.0]]),
            "g_scalar_like": torch.tensor([70.0]),
            "g_scalar": torch.tensor(80.0),
            "g_high": (torch.arange(6.0) + 100.0).reshape(1, 1, 2, 3),
            "g_no_compress": (torch.arange(6.0) + 200.0).reshape(2, 1, 3),
        },
    ]

    result = collate_graph_samples(batch_list)

    assert result["g_vec"].shape == (2, 3)
    assert result["g_row"].shape == (2, 3)
    assert result["g_scalar_like"].shape == (2,)
    assert result["g_scalar"].shape == (2,)
    assert result["g_high"].shape == (2, 2, 3)
    assert result["g_no_compress"].shape == (2, 2, 1, 3)


def test_graph_attr_normalize_numpy_with_multiple_leading_singletons():
    """Validate numpy graph attributes with multiple leading singleton dims are normalized correctly."""
    batch_list = [
        {
            "num_nodes": 2,
            "x": torch.tensor([[1.0], [2.0]]),
            "np_multi": np.arange(6, dtype=np.float32).reshape(1, 1, 2, 3),
            "np_scalar_like": np.array([[[7.0]]], dtype=np.float32),
        },
        {
            "num_nodes": 1,
            "x": torch.tensor([[3.0]]),
            "np_multi": (np.arange(6, dtype=np.float32) + 100).reshape(1, 1, 2, 3),
            "np_scalar_like": np.array([[[8.0]]], dtype=np.float32),
        },
    ]

    result = collate_graph_samples(batch_list)

    assert result["np_multi"].shape == (2, 2, 3)
    assert result["np_scalar_like"].shape == (2,)
    assert torch.equal(result["np_scalar_like"], torch.tensor([7.0, 8.0]))


def test_no_anchor_default_multiple_variable_keys_same_signature():
    """Validate stable fallback for multiple no-anchor variable keys sharing signatures."""
    batch_list = [
        {
            "x": torch.tensor([[1.0], [2.0]]),
            "z": torch.tensor([[7.0, 8.0], [9.0, 10.0]]),
            "graph_attr": torch.tensor([1.0]),
        },
        {
            "x": torch.tensor([[3.0], [4.0], [5.0], [6.0]]),
            "z": torch.tensor([[11.0, 12.0], [13.0, 14.0], [15.0, 16.0], [17.0, 18.0]]),
            "graph_attr": torch.tensor([2.0]),
        },
    ]

    result = collate_graph_samples(batch_list)

    assert torch.equal(result["batch"], torch.tensor([0, 0, 1, 1, 1, 1]))
    assert result["x"].shape == (6, 1)
    assert result["z"].shape == (6, 2)
    assert torch.equal(result["x"][:2], torch.tensor([[1.0], [2.0]]))
    assert torch.equal(result["x"][2:], torch.tensor([[3.0], [4.0], [5.0], [6.0]]))
    assert torch.equal(result["graph_attr"], torch.tensor([1.0, 2.0]))


def test_no_anchor_strict_raises_for_mixed_attrs():
    """Validate strict no-anchor mode raises on unresolved variable-length attributes."""
    batch_list = [
        {
            "x": torch.randn(2, 3),
            "graph_attr": torch.tensor([1.0]),
        },
        {
            "x": torch.randn(4, 3),
            "graph_attr": torch.tensor([2.0]),
        },
    ]

    with pytest.raises(ValueError, match="Cannot determine attribute types"):
        determine_graph_key_types(batch_list, strict=True)


def test_determine_graph_key_types_strict_raises_when_variable_keys_cannot_be_disambiguated():
    """Validate strict mode raises when variable-length attributes have no anchors."""
    batch_list = [
        {
            "x": torch.randn(3, 4),
            "aux": torch.randn(3, 2),
            "graph_attr": torch.tensor([1.0]),
        },
        {
            "x": torch.randn(5, 4),
            "aux": torch.randn(5, 2),
            "graph_attr": torch.tensor([2.0]),
        },
    ]

    with pytest.raises(ValueError, match="Cannot determine attribute types"):
        determine_graph_key_types(batch_list, strict=True)


# Section: Specialized Edge Keys
# Validate naming-convention keys (edge_*_index / edge_*_attr), offsets, concatenation, and validation errors.


def test_special_named_edge_index_offsets_like_edge_index():
    """Validate named edge-index keys apply node-offset collation like edge_index."""
    batch_list = [
        {
            "num_nodes": 3,
            "x": torch.randn(3, 2),
            "edge_d_index": torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
            "graph_attr": torch.tensor([1.0]),
        },
        {
            "num_nodes": 2,
            "x": torch.randn(2, 2),
            "edge_d_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            "graph_attr": torch.tensor([2.0]),
        },
    ]

    result = collate_graph_samples(batch_list)

    assert "edge_d_index" in result
    expected = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 0, 4, 3]], dtype=torch.long)
    assert torch.equal(result["edge_d_index"], expected)


@pytest.mark.parametrize(
    "batch_list",
    [
        pytest.param(
            [
                {
                    "edge_d_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
                    "graph_attr": torch.tensor([1.0]),
                },
                {
                    "edge_d_index": torch.tensor([[0], [2]], dtype=torch.long),
                    "graph_attr": torch.tensor([2.0]),
                },
            ],
            id="edge_index_without_node_hints",
        ),
        pytest.param(
            [
                {
                    "edge_d_attr": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                    "graph_attr": torch.tensor([1.0]),
                },
                {
                    "edge_d_attr": torch.tensor([[5.0, 6.0]]),
                    "graph_attr": torch.tensor([2.0]),
                },
            ],
            id="edge_attr_without_node_hints",
        ),
    ],
)
def test_special_named_edge_keys_without_node_hints_raises(batch_list):
    """Validate specialized edge keys raise when no node hints are available."""
    with pytest.raises(ValueError, match="specialized edge keys were provided"):
        collate_graph_samples(batch_list)


@pytest.mark.parametrize(
    "batch_list, expected_shape, expected_prefix",
    [
        pytest.param(
            [
                {
                    "num_nodes": 2,
                    "x": torch.randn(2, 1),
                    "edge_d_attr": torch.tensor([0.1, 0.2, 0.3, 0.4]),
                    "graph_attr": torch.tensor([1.0]),
                },
                {
                    "num_nodes": 3,
                    "x": torch.randn(3, 1),
                    "edge_d_attr": torch.tensor([0.5] * 9),
                    "graph_attr": torch.tensor([2.0]),
                },
            ],
            (13,),
            torch.tensor([0.1, 0.2, 0.3, 0.4]),
            id="edge_attr_1d_concat",
        ),
        pytest.param(
            [
                {
                    "num_nodes": 2,
                    "x": torch.randn(2, 1),
                    "edge_d_attr": torch.randn(4, 3),
                    "graph_attr": torch.tensor([1.0]),
                },
                {
                    "num_nodes": 3,
                    "x": torch.randn(3, 1),
                    "edge_d_attr": torch.randn(9, 3),
                    "graph_attr": torch.tensor([2.0]),
                },
            ],
            (13, 3),
            None,
            id="edge_attr_2d_concat",
        ),
    ],
)
def test_special_named_edge_attr_concatenate(batch_list, expected_shape, expected_prefix):
    """Validate specialized edge attributes concatenate along dim=0 for both 1D and 2D inputs."""
    result = collate_graph_samples(batch_list)
    assert "edge_d_attr" in result
    assert result["edge_d_attr"].shape == expected_shape
    if expected_prefix is not None:
        assert torch.allclose(result["edge_d_attr"][: expected_prefix.shape[0]], expected_prefix)


def test_special_named_edge_index_and_attr_together():
    """Validate coexisting specialized edge index and attr keys collate correctly."""
    batch_list = [
        {
            "num_nodes": 2,
            "x": torch.randn(2, 2),
            "edge_d_index": torch.tensor([[0, 1, 0, 1], [0, 0, 1, 1]], dtype=torch.long),
            "edge_d_attr": torch.arange(4, dtype=torch.float32),
            "graph_attr": torch.tensor([1.0]),
        },
        {
            "num_nodes": 3,
            "x": torch.randn(3, 2),
            "edge_d_index": torch.tensor([[0, 1, 2, 0, 1, 2, 0, 1, 2], [0, 0, 0, 1, 1, 1, 2, 2, 2]], dtype=torch.long),
            "edge_d_attr": torch.arange(9, dtype=torch.float32) + 10,
            "graph_attr": torch.tensor([2.0]),
        },
    ]

    result = collate_graph_samples(batch_list)

    assert result["edge_d_index"].shape == (2, 13)
    assert result["edge_d_attr"].shape == (13,)
    # First 4 attrs come from sample0, next 9 from sample1
    assert torch.equal(result["edge_d_attr"][:4], torch.arange(4, dtype=torch.float32))
    assert torch.equal(result["edge_d_attr"][4:], torch.arange(9, dtype=torch.float32) + 10)


@pytest.mark.parametrize(
    "batch_list, expected_exception, expected_match",
    [
        pytest.param(
            [
                {
                    "num_nodes": 2,
                    "x": torch.randn(2, 1),
                    "edge_d_index": torch.tensor([0, 1, 1, 0], dtype=torch.long),
                    "graph_attr": torch.tensor([1.0]),
                }
            ],
            ValueError,
            r"must have shape \[2, X\]",
            id="edge_index_invalid_shape",
        ),
        pytest.param(
            [
                {
                    "num_nodes": 2,
                    "x": torch.randn(2, 1),
                    "edge_d_index": torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32),
                    "graph_attr": torch.tensor([1.0]),
                }
            ],
            TypeError,
            "must be an integer tensor",
            id="edge_index_invalid_dtype",
        ),
        pytest.param(
            [
                {
                    "num_nodes": 2,
                    "x": torch.randn(2, 1),
                    "edge_d_attr": torch.tensor(1.0),
                    "graph_attr": torch.tensor([1.0]),
                }
            ],
            ValueError,
            "must have at least 1 dimension",
            id="edge_attr_invalid_scalar",
        ),
    ],
)
def test_special_named_edge_keys_invalid_inputs_raise(batch_list, expected_exception, expected_match):
    """Validate invalid specialized edge key inputs raise expected exceptions."""
    with pytest.raises(expected_exception, match=expected_match):
        collate_graph_samples(batch_list)


def test_edge_index_and_named_edge_index_coexist_with_correct_offsets():
    """Validate standard and named edge-index keys coexist with correct offsets."""
    batch_list = [
        {
            "num_nodes": 3,
            "x": torch.randn(3, 1),
            "edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            "edge_d_index": torch.tensor([[0, 2, 1], [2, 1, 0]], dtype=torch.long),
            "graph_attr": torch.tensor([1.0]),
        },
        {
            "num_nodes": 2,
            "x": torch.randn(2, 1),
            "edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            "edge_d_index": torch.tensor([[0, 1], [0, 1]], dtype=torch.long),
            "graph_attr": torch.tensor([2.0]),
        },
    ]

    result = collate_graph_samples(batch_list)

    expected_edge_index = torch.tensor([[0, 1, 3, 4], [1, 2, 4, 3]], dtype=torch.long)
    expected_edge_d_index = torch.tensor([[0, 2, 1, 3, 4], [2, 1, 0, 3, 4]], dtype=torch.long)
    assert torch.equal(result["edge_index"], expected_edge_index)
    assert torch.equal(result["edge_d_index"], expected_edge_d_index)


def test_named_edge_attr_keeps_order_with_coexisting_edge_index():
    """Validate specialized edge-attr order is preserved with standard edge_index."""
    batch_list = [
        {
            "num_nodes": 2,
            "x": torch.randn(2, 1),
            "edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            "edge_d_index": torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=torch.long),
            "edge_d_attr": torch.tensor([1.0, 2.0, 3.0, 4.0]),
            "graph_attr": torch.tensor([1.0]),
        },
        {
            "num_nodes": 3,
            "x": torch.randn(3, 1),
            "edge_index": torch.tensor([[0, 2], [2, 1]], dtype=torch.long),
            "edge_d_index": torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long),
            "edge_d_attr": torch.tensor([10.0, 20.0, 30.0]),
            "graph_attr": torch.tensor([2.0]),
        },
    ]

    result = collate_graph_samples(batch_list)
    assert torch.equal(result["edge_d_attr"], torch.tensor([1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0]))


def test_multiple_named_edge_index_keys_are_independently_collated():
    """Validate multiple named edge-index keys are collated independently."""
    batch_list = [
        {
            "num_nodes": 3,
            "x": torch.randn(3, 1),
            "edge_d_index": torch.tensor([[0, 1], [2, 0]], dtype=torch.long),
            "edge_k_index": torch.tensor([[0, 2], [1, 1]], dtype=torch.long),
            "graph_attr": torch.tensor([1.0]),
        },
        {
            "num_nodes": 2,
            "x": torch.randn(2, 1),
            "edge_d_index": torch.tensor([[0], [1]], dtype=torch.long),
            "edge_k_index": torch.tensor([[1], [0]], dtype=torch.long),
            "graph_attr": torch.tensor([2.0]),
        },
    ]

    result = collate_graph_samples(batch_list)

    expected_d = torch.tensor([[0, 1, 3], [2, 0, 4]], dtype=torch.long)
    expected_k = torch.tensor([[0, 2, 4], [1, 1, 3]], dtype=torch.long)
    assert torch.equal(result["edge_d_index"], expected_d)
    assert torch.equal(result["edge_k_index"], expected_k)


def test_plain_edge_attr_not_treated_as_named_special_key():
    """Validate plain edge_attr stays on the generic edge-attribute path."""
    batch_list = [
        {
            "num_nodes": 2,
            "x": torch.randn(2, 1),
            "edge_attr": torch.tensor([[1.0], [2.0]]),
            "graph_attr": torch.tensor([1.0]),
        },
        {
            "num_nodes": 3,
            "x": torch.randn(3, 1),
            "edge_attr": torch.tensor([[3.0], [4.0], [5.0]]),
            "graph_attr": torch.tensor([2.0]),
        },
    ]

    result = collate_graph_samples(batch_list)
    # edge_attr stays on the original path and should still concatenate as edge attribute.
    assert result["edge_attr"].shape == (5, 1)
    assert torch.equal(result["edge_attr"], torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]]))


def test_collate_graph_samples_with_explicit_key_types_keeps_non_tensor_attrs():
    """Validate explicit key_types does not require non-tensor attrs to be pre-classified."""
    batch_list = [
        {
            "num_nodes": 2,
            "x": torch.tensor([[1.0], [2.0]]),
            "graph_note": "first",
            "graph_meta": {"split": "train", "id": 1},
        },
        {
            "num_nodes": 1,
            "x": torch.tensor([[3.0]]),
            "graph_note": "second",
            "graph_meta": {"split": "val", "id": 2},
        },
    ]
    key_types = {"node": {"x"}, "edge": set(), "graph": set()}

    result = collate_graph_samples(batch_list, key_types=key_types)

    assert torch.equal(result["batch"], torch.tensor([0, 0, 1]))
    assert torch.equal(result["x"], torch.tensor([[1.0], [2.0], [3.0]]))
    assert result["graph_note"] == ["first", "second"]
    assert result["graph_meta"] == [
        {"split": "train", "id": 1},
        {"split": "val", "id": 2},
    ]


def test_collate_graph_samples_with_explicit_key_types_requires_tensor_classification():
    """Validate explicit key_types still enforces classification coverage for tensor attrs."""
    batch_list = [
        {
            "num_nodes": 2,
            "x": torch.tensor([[1.0], [2.0]]),
            "aux": torch.tensor([[10.0], [20.0]]),
            "graph_note": "first",
        },
        {
            "num_nodes": 1,
            "x": torch.tensor([[3.0]]),
            "aux": torch.tensor([[30.0]]),
            "graph_note": "second",
        },
    ]
    key_types = {"node": {"x"}, "edge": set(), "graph": set()}

    with pytest.raises(ValueError, match="not classified"):
        collate_graph_samples(batch_list, key_types=key_types)
