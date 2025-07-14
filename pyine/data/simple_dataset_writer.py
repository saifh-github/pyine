import pyine.data.lmdb_io

parser = pyine.data.lmdb_io.LMDBReader(
    path="data/2025-03-31-v01-lmdb",
)

problem_count = len(parser)
print(f"{problem_count=}")
metadata = parser.get_metadata()
print(f"{metadata=}")

for problem_idx, problem_data in enumerate(parser):
    print(f"{problem_idx=}")
    assert isinstance(problem_data, dict)
    print(f"{problem_data=}")
    break
