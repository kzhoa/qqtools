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


def test_empty_batch():
    """Test collating an empty list."""
    batch_list = []
    result = collate_graph_samples(batch_list)
    assert result == {}


def test_single_graph_no_edge_index():
    """Test collating a single graph without edge_index but with num_nodes."""
    nodes = 3
    sample = create_sample(num_nodes=nodes, edge_index=None, node_features=[[1], [2], [3]], graph_features=[10])
    batch_list = [sample]
    result = collate_graph_samples(batch_list)

    assert "batch" in result
    assert torch.equal(result["batch"], torch.tensor([0, 0, 0]))
    assert "x" in result
    assert torch.equal(result["x"], torch.tensor([[1], [2], [3]]))
    assert "graph_attr" in result
    assert torch.equal(result["graph_attr"], torch.tensor([[10]]))
    assert "num_nodes" not in result  # num_nodes is usually removed after collation


def test_single_graph_with_edge_index():
    """Test collating a single graph with edge_index."""
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
    assert torch.equal(result["graph_attr"], torch.tensor([[100]]))
    assert "edge_index" in result
    assert torch.equal(result["edge_index"], torch.tensor(edge_index))
    assert "num_nodes" not in result


def test_two_graphs_different_nodes_no_edge_index():
    """Test two graphs, varying nodes, no edge_index, relying on num_nodes."""
    sample1 = create_sample(num_nodes=3, edge_index=None, node_features=[[1], [2], [3]], graph_features=[10])
    sample2 = create_sample(num_nodes=2, edge_index=None, node_features=[[4], [5]], graph_features=[20])
    batch_list = [sample1, sample2]
    result = collate_graph_samples(batch_list)

    assert "batch" in result
    assert torch.equal(result["batch"], torch.tensor([0, 0, 0, 1, 1]))
    assert "x" in result
    assert torch.equal(result["x"], torch.tensor([[1], [2], [3], [4], [5]]))
    assert "graph_attr" in result
    assert torch.equal(result["graph_attr"], torch.tensor([[10], [20]]))  # Stacked graph attrs
    assert "edge_index" not in result


def test_two_graphs_different_nodes_with_edge_index():
    """Test two graphs, varying nodes, with edge_index."""
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
    assert torch.equal(result["graph_attr"], torch.tensor([[100], [200]]))
    assert "edge_index" in result
    # Expected edge_index: first graph's [0,1] shifted by 0, second graph's [0,1] shifted by 3
    expected_edge_index = torch.tensor([[0, 1, 3, 4], [1, 2, 4, 3]])
    assert torch.equal(result["edge_index"], expected_edge_index)


def test_graphs_with_varying_edge_attributes():
    """Test graphs where edge attributes might be misidentified if not careful."""
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
    )  # Same number of edges as sample1
    batch_list = [sample1, sample2]
    result = collate_graph_samples(batch_list)

    assert "edge_attr" in result
    assert torch.equal(result["edge_attr"], torch.tensor([[10], [11], [20], [21]]))


def test_graphs_with_only_graph_attributes_and_num_nodes():
    """Test graphs with only graph attributes and num_nodes specified."""
    sample1 = {"num_nodes": 2, "graph_attr": torch.tensor([10.0])}
    sample2 = {"num_nodes": 3, "graph_attr": torch.tensor([20.0])}
    batch_list = [sample1, sample2]

    # Manually specify key types. Node/edge sets are empty.
    key_types = {"graph": {"graph_attr"}, "node": set(), "edge": set()}
    result = collate_graph_samples(batch_list, key_types=key_types)

    assert "batch" in result
    assert torch.equal(result["batch"], torch.tensor([0, 0, 1, 1, 1]))

    assert "graph_attr" in result
    assert torch.equal(result["graph_attr"], torch.tensor([[10.0], [20.0]]))  # Corrected assertion for 1D tensor output

    assert "edge_index" not in result

    assert "x" not in result
    assert "dummy_node_attr" not in result

    assert "num_nodes" not in result


def test_graphs_with_edge_index_but_no_node_attributes():
    """Test when edge_index is present, inferring nodes, but no explicit node attributes."""
    sample1 = {"edge_index": torch.tensor([[0, 1], [1, 0]]), "graph_attr": torch.tensor([10.0])}  # Implies 2 nodes
    sample2 = {"edge_index": torch.tensor([[0], [0]]), "graph_attr": torch.tensor([20.0])}  # Implies 1 node
    batch_list = [sample1, sample2]

    result = collate_graph_samples(batch_list)

    assert "batch" in result
    # Graph 1 has 2 nodes, Graph 2 has 1 node. Total 3 nodes. Indices: [0, 0] for G1, [1] for G2.
    assert torch.equal(result["batch"], torch.tensor([0, 0, 1]))
    assert "graph_attr" in result
    # Assumes graph attributes are stacked.
    assert torch.equal(result["graph_attr"], torch.tensor([[10.0], [20.0]]))
    assert "edge_index" in result
    # Expected edge_index: G1 edges [[0, 1], [1, 0]], G2 edges shifted [[2], [2]]
    # Combined: [[0, 1, 2], [1, 0, 2]]
    print(result["edge_index"])  # Added print for debugging if needed
    assert torch.equal(result["edge_index"], torch.tensor([[0, 1, 2], [1, 0, 2]]))
    assert "x" not in result
    assert "num_nodes" not in result


def test_graphs_with_ambiguous_shape_constant_but_different_meaning():
    """
    Test case where an attribute has constant shape across batch but should be node attr.
    This relies on the `is_shape_constant` check within determine_graph_key_types.
    """
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
    assert torch.equal(result["graph_attr"], torch.tensor([[10], [20]]))  # Stacked graph attrs
    assert "edge_index" in result
    assert torch.equal(result["edge_index"], torch.tensor([[0, 1, 3, 4], [2, 0, 4, 5]]))


def test_graph_attribute_misclassification_recovery():
    """
    Test case simulating misclassification of a node attribute as graph attribute
    and verifying it can be recovered (though the current implementation aims to prevent this).
    This test asserts the *expected* behavior (correct classification) rather than
    testing the recovery mechanism directly, as the goal is correct initial classification.
    """
    # Scenario: 2 graphs, both with 3 nodes. Node feature 'x' is [3, 10]. Graph feature 'g' is [].
    # If 'x' is mistakenly classified as graph attribute (because shape is constant [3, 10] across batch)
    # it would be stacked to [2, 3, 10].
    # The correct classification should make it node attribute, concatenated to [6, 10].

    sample1 = create_sample(
        num_nodes=3, edge_index=[[0, 1], [2, 0]], node_features=[[1.0] * 10] * 3, graph_features=[10.0]
    )
    sample2 = create_sample(
        num_nodes=3, edge_index=[[0, 1], [1, 2]], node_features=[[2.0] * 10] * 3, graph_features=[20.0]
    )
    batch_list = [sample1, sample2]

    result = collate_graph_samples(batch_list)

    # Assert that 'x' is treated as a node attribute (concatenated)
    assert "x" in result
    assert result["x"].shape == (6, 10)  # Should be (num_nodes1 + num_nodes2, feature_dim)
    assert torch.equal(result["x"][:3, :], torch.tensor([[1.0] * 10] * 3))
    assert torch.equal(result["x"][3:, :], torch.tensor([[2.0] * 10] * 3))

    # Assert 'graph_attr' is treated as a graph attribute (stacked)
    assert "graph_attr" in result
    assert result["graph_attr"].shape == (2, 1)  # Should be (num_graphs,)
    assert torch.equal(result["graph_attr"], torch.tensor([[10.0], [20.0]]))


def test_edge_case_empty_edge_index():
    """Test when edge_index is present but empty."""
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
    """Test when a node attribute is missing in one sample (should ideally error or handle gracefully)."""
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


def test_determine_graph_key_types_raises_when_variable_keys_cannot_be_disambiguated():
    """Variable-length attributes without node/edge anchors should raise explicit error."""
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
        determine_graph_key_types(batch_list)


def test_special_named_edge_index_offsets_like_edge_index():
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


def test_special_named_edge_index_infers_num_nodes_without_edge_index():
    batch_list = [
        {
            "edge_d_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            "graph_attr": torch.tensor([1.0]),
        },
        {
            "edge_d_index": torch.tensor([[0], [2]], dtype=torch.long),
            "graph_attr": torch.tensor([2.0]),
        },
    ]

    result = collate_graph_samples(batch_list)

    # sample0 infers 2 nodes, sample1 infers 3 nodes
    assert torch.equal(result["batch"], torch.tensor([0, 0, 1, 1, 1]))
    expected = torch.tensor([[0, 1, 2], [1, 0, 4]], dtype=torch.long)
    assert torch.equal(result["edge_d_index"], expected)


def test_special_named_edge_attr_concatenate_1d():
    batch_list = [
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
    ]

    result = collate_graph_samples(batch_list)

    assert "edge_d_attr" in result
    assert result["edge_d_attr"].shape == (13,)
    assert torch.allclose(result["edge_d_attr"][:4], torch.tensor([0.1, 0.2, 0.3, 0.4]))


def test_special_named_edge_attr_concatenate_2d():
    batch_list = [
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
    ]

    result = collate_graph_samples(batch_list)
    assert result["edge_d_attr"].shape == (13, 3)


def test_special_named_edge_index_and_attr_together():
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


def test_special_named_edge_index_invalid_shape_raises():
    batch_list = [
        {
            "num_nodes": 2,
            "x": torch.randn(2, 1),
            "edge_d_index": torch.tensor([0, 1, 1, 0], dtype=torch.long),
            "graph_attr": torch.tensor([1.0]),
        }
    ]

    with pytest.raises(ValueError, match=r"must have shape \[2, E\]"):
        collate_graph_samples(batch_list)


def test_special_named_edge_index_invalid_dtype_raises():
    batch_list = [
        {
            "num_nodes": 2,
            "x": torch.randn(2, 1),
            "edge_d_index": torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32),
            "graph_attr": torch.tensor([1.0]),
        }
    ]

    with pytest.raises(TypeError, match="must be an integer tensor"):
        collate_graph_samples(batch_list)


def test_special_named_edge_attr_invalid_scalar_raises():
    batch_list = [
        {
            "num_nodes": 2,
            "x": torch.randn(2, 1),
            "edge_d_attr": torch.tensor(1.0),
            "graph_attr": torch.tensor([1.0]),
        }
    ]

    with pytest.raises(ValueError, match="must have at least 1 dimension"):
        collate_graph_samples(batch_list)
