import typing

import torch.utils.data

import pyine.data.lmdb_io


class DatasetParser(torch.utils.data.Dataset):
    """PyINE raw dataset reader.

    Note: this readers allows access to the RAW traces along with the original code
    and related JSON data. It does NOT attempt to structure the traces into anything
    useful for explanation-related experiments.

    Args:
        lmdb_path: Path to the LMDB database containing code traces.
    """

    def __init__(
        self,
        lmdb_path: str,
    ) -> None:
        super().__init__()
        self.lmdb_path = lmdb_path
        self.reader = pyine.data.lmdb_io.LMDBReader(path=self.lmdb_path)

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Total number of elements in the LMDB database.
        """
        return len(self.reader)

    def __getitem__(self, index: int) -> dict[str, typing.Any]:
        """
        Fetches an individual sample from the LMDB database by its index.

        Args:
            index: Index of the sample to retrieve.
        """
        return self.reader.get(index)

    def close(self) -> None:
        """
        Closes the LMDBReader instance and releases resources.
        """
        self.reader.close()


if __name__ == "__main__":
    _dataset_reader = DatasetParser(lmdb_path="data/2025-03-31-v01.raw.lmdb")
    print(f"dataset contains {len(_dataset_reader)} samples")
    _sample = _dataset_reader[0]
    print(f"sample: {_sample}")
    _dataset_reader.close()
