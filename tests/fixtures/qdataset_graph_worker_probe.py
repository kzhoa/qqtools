import sys

import torch

from qqtools.torch.qdataset import qDictDataloader, qDictDataset


class _GraphDataset(qDictDataset):
    pass


def main(start_method: str) -> None:
    dataset = _GraphDataset(
        data_list=[
            {
                "num_nodes": 2,
                "x": torch.tensor([[1.0], [2.0]]),
                "edge_index": torch.tensor([[0], [1]]),
            },
            {
                "num_nodes": 3,
                "x": torch.tensor([[3.0], [4.0], [5.0]]),
                "edge_index": torch.tensor([[0, 1], [1, 2]]),
            },
        ]
    )
    loader = qDictDataloader(
        dataset=dataset,
        batch_size=2,
        is_graph=True,
        num_workers=1,
        multiprocessing_context=start_method,
    )

    batch = next(iter(loader))

    assert torch.equal(batch["batch"], torch.tensor([0, 0, 1, 1, 1]))
    assert torch.equal(batch["edge_index"], torch.tensor([[0, 2, 3], [1, 3, 4]]))


if __name__ == "__main__":
    main(sys.argv[1])
