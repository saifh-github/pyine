"""
Configurable fake dataset readers for unit tests.

This module provides fakes for:
- pyine.data.traces.dataset_reader.DatasetReader
- pyine.data.deltas.dataset_reader.DatasetReader

They generate realistic-enough data structures (TraceResult, CodingProblem, TraceDeltaList) to
avoid dumb mocks while remaining lightweight and fully in-memory. They can be used directly or
monkeypatched to replace real readers in unit tests.

Example usage (direct):
    >>> from tests.utils.fake_dataset_readers import FakeTraceDatasetReader
    >>>
    >>> reader = FakeTraceDatasetReader(num_problems=2, solutions_per_problem=1, tests_per_problem=2)
    >>> assert len(reader) > 0
    >>> trace = reader[0]  # pyine.utils.code.execution.TraceResult
    >>> problem = reader.get_problem_data(0)  # pyine.data.traces.dataset_utils.CodingProblem

Example usage (pytest + monkeypatch):
    >>> import pytest
    >>>
    >>> @pytest.fixture()
    >>> def patch_trace_reader(monkeypatch):
    >>>     from tests.utils.fake_dataset_readers import FakeTraceDatasetReader
    >>>     import pyine.data.traces.dataset_reader as traces_dr
    >>>     monkeypatch.setattr(traces_dr, "DatasetReader", FakeTraceDatasetReader)
    >>>     return FakeTraceDatasetReader
    >>>
    >>> @pytest.fixture()
    >>> def patch_deltas_reader(monkeypatch):
    >>>     from tests.utils.fake_dataset_readers import FakeDeltaDatasetReader
    >>>     import pyine.data.deltas.dataset_reader as deltas_dr
    >>>     monkeypatch.setattr(deltas_dr, "DatasetReader", FakeDeltaDatasetReader)
    >>>     return FakeDeltaDatasetReader
"""

import dataclasses
import datetime
import functools
import hashlib
import random
import typing

import pyine.data.deltas.dataset_utils as deltas_utils
import pyine.data.traces.dataset_utils as traces_utils
import pyine.utils.code.execution as exec_utils


@dataclasses.dataclass
class FakeTraceDataConfig:
    """Configuration for the fake trace reader.

    All parameters are deterministic given the seed and control how much data is generated.
    """

    dataset_name: str = "FAKE"
    subset_name: str = "unit"
    num_problems: int = 2
    solutions_per_problem: int = 1
    tests_per_problem: int = 1
    augmented_per_solution: int = 0  # number of augmented variants per solution/test
    entrypoint_name: str | None = "solution"
    max_valid_events: int | None = None
    max_events_per_line: int | None = None
    max_var_repr_length: int | None = None
    seed: int = 123
    code_kind: str = "function"  # "function" or "script"
    # note: for function kind, inputs will be passed as a single param to the entrypoint


class _FakeBase:

    def get_metadata(self) -> dict[str, typing.Any]:  # pragma: no cover - trivial
        return {
            "parent_dataset": {
                "dataset_name": getattr(self, "_dataset_name", "FAKE"),
                "subset_name": getattr(self, "_subset_name", "unit"),
                "trace_count": len(self),  # noqa
                "generated_at": datetime.datetime.now().isoformat(),
            }
        }

    def get_parent_dataset_name(self) -> str:  # pragma: no cover - trivial
        return self.get_metadata()["parent_dataset"]["dataset_name"]

    def get_size_on_disk(self) -> int:  # pragma: no cover - trivial
        return 0

    def get_hash(self) -> str:  # pragma: no cover - trivial
        hsrc = f"{self.get_parent_dataset_name()}::{len(self)}::{self.__class__.__name__}"  # noqa
        return hashlib.sha256(hsrc.encode()).hexdigest()[:16]

    def close(self) -> None:  # pragma: no cover - trivial
        return None


class FakeTraceDatasetReader(_FakeBase):
    """Fake replacement for pyine.data.traces.dataset_reader.DatasetReader.

    This class mimics the core interface of the real reader while generating realistic
    TraceResult objects by actually executing minimal Python code and tracing it.

    Constructor signature accepts the same first positional arg (lmdb_path) but ignores it,
    and supports additional configuration through kwargs or a `config` object.
    """

    def __init__(
        self,
        lmdb_path: typing.Any | None = None,
        *,
        config: FakeTraceDataConfig | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        # merge config from kwargs if provided
        self._cfg = config or FakeTraceDataConfig(
            **{k: v for k, v in kwargs.items() if k in {f.name for f in dataclasses.fields(FakeTraceDataConfig)}}
        )
        self._dataset_name = self._cfg.dataset_name
        self._subset_name = self._cfg.subset_name
        self._rng = random.Random(self._cfg.seed)
        # public attributes mirroring real reader
        self.problem_keys: list[str] = []
        self.trace_keys: list[str] = []
        self.trace_key_to_problem_key: dict[str, str] = {}
        self.augment_key_to_parent_trace_key: dict[str, str] = {}
        # private attributes with INTERNAL indices
        self._problem_indices: list[int] = []
        self._trace_indices: list[int] = []
        self._trace_idx_to_problem_idx: dict[int, int] = {}
        self._augment_idx_to_parent_trace_idx: dict[int, int] = {}
        # internal storages
        self._problems: list[traces_utils.CodingProblem] = []
        self._traces: list[exec_utils.TraceResult] = []
        # generate problems and traces
        self._generate_data()

    # ---------------------------- public API ----------------------------
    def __len__(self) -> int:
        return len(self.trace_keys)

    def __getitem__(self, index_or_key: int | str) -> exec_utils.TraceResult:
        idx = self._resolve_index(index_or_key)
        return self._traces[idx]

    @functools.lru_cache(maxsize=512)
    def get_problem_data(self, index_or_key: int | str) -> traces_utils.CodingProblem:
        idx = self._resolve_index(index_or_key)
        problem_idx = self._trace_idx_to_problem_idx[self._trace_indices[idx]]
        # internal problem_idx is already an integer index into self._problems
        return self._problems[problem_idx]

    def get_tags(self, index_or_key: int | str) -> list[str]:
        idx = self._resolve_index(index_or_key)
        problem_idx = self._trace_idx_to_problem_idx[self._trace_indices[idx]]
        return self._problems[problem_idx].problem_tags

    # ---------------------------- internals ----------------------------
    def _resolve_index(self, index_or_key: int | str) -> int:
        if isinstance(index_or_key, int):
            if not (0 <= index_or_key < len(self)):
                raise IndexError(f"index {index_or_key} out of range")
            return index_or_key
        elif isinstance(index_or_key, str):
            if index_or_key not in self.trace_keys:
                raise KeyError(f"key {index_or_key} not found in fake dataset")
            return self.trace_keys.index(index_or_key)
        else:  # pragma: no cover
            raise ValueError(f"invalid index_or_key type: {type(index_or_key)}")

    def _generate_data(self) -> None:
        # prepare problems
        for p_idx in range(self._cfg.num_problems):
            cp = self._make_problem(p_idx)
            self._problems.append(cp)
            self._problem_indices.append(p_idx)
            pkey = f"{cp.problem_id}{traces_utils.PROBLEM_DATA_SUFFIX}"
            self.problem_keys.append(pkey)
        # generate traces per problem/solution/test (+augmentations)
        for p_idx, problem in enumerate(self._problems):
            for s_idx in range(self._cfg.solutions_per_problem):
                for t_idx in range(self._cfg.tests_per_problem):
                    # base trace
                    base_trace, base_key = self._make_trace(problem, s_idx, t_idx, augment=None)
                    self._append_trace(base_trace, base_key, p_idx)
                    # augmentations
                    for a_idx in range(self._cfg.augmented_per_solution):
                        augm_tag = ("simple", a_idx)
                        aug_trace, aug_key = self._make_trace(problem, s_idx, t_idx, augment=augm_tag)
                        self._append_trace(aug_trace, aug_key, p_idx, parent_key=base_key)

        assert len(self._trace_indices) == len(self.trace_keys) == len(self._traces)

    def _append_trace(
        self,
        trace: exec_utils.TraceResult,
        tkey: str,
        problem_idx: int,
        parent_key: str | None = None,
    ) -> None:
        internal_idx = len(self._traces)
        self._traces.append(trace)
        self._trace_indices.append(internal_idx)
        self.trace_keys.append(tkey)
        self._trace_idx_to_problem_idx[internal_idx] = problem_idx
        pkey = self.problem_keys[problem_idx]
        self.trace_key_to_problem_key[tkey] = pkey
        if parent_key is not None:
            self._augment_idx_to_parent_trace_idx[internal_idx] = self._trace_indices[self.trace_keys.index(parent_key)]
            self.augment_key_to_parent_trace_key[tkey] = parent_key

    def _make_problem(self, p_idx: int) -> traces_utils.CodingProblem:
        pid = traces_utils.CodingProblemIdentifier(
            dataset=self._dataset_name,
            subset=self._subset_name,
            problem_idx=p_idx,
        )
        # create deterministic test pairs (inputs, expected outputs unused but present)
        tests: list[tuple] = []
        for i in range(self._cfg.tests_per_problem):
            inp = (p_idx + 1) * (i + 2)  # deterministic int input
            out = inp * 2  # pretend expected output
            tests.append((inp, out))
        solutions = [
            traces_utils.SolutionIdentifier(
                dataset=self._dataset_name, subset=self._subset_name, problem_idx=p_idx, solution_idx=s_idx
            )
            for s_idx in range(self._cfg.solutions_per_problem)
        ]
        cp = traces_utils.CodingProblem(
            source_dataset_name=self._dataset_name,
            source_data_path=f"/{self._dataset_name}/{self._subset_name}",
            source_data_hash=hashlib.sha1(f"{pid}".encode()).hexdigest()[:16],
            problem_id=pid,
            problem_statement=f"Compute f(x) = 2*x for problem {p_idx}",
            problem_tags=["math", "toy"],
            test_inout_pairs=tests,
            entrypoint_name=self._cfg.entrypoint_name,
            potential_solution_ids=solutions,
            parsing_errors=None,
            is_banned=False,
        )
        return cp

    def _make_trace(
        self,
        problem: traces_utils.CodingProblem,
        s_idx: int,
        t_idx: int,
        augment: tuple[str, int] | None,
    ) -> tuple[exec_utils.TraceResult, str]:
        # identifier + key
        tid = traces_utils.TraceIdentifier(
            dataset=problem.problem_id.dataset,
            subset=problem.problem_id.subset,
            problem_idx=problem.problem_id.problem_idx,
            solution_idx=s_idx,
            test_idx=t_idx,
            augment_category=(augment[0] if augment else None),
            augment_idx=(augment[1] if augment else None),
        )
        key = str(tid)
        code = self._gen_code(problem, s_idx)
        inputs = problem.test_inout_pairs[t_idx][0]
        trace_res = exec_utils.execute_and_trace_code(
            code_string=code,
            inputs=inputs,
            identifier=str(tid),
            entrypoint_name=self._cfg.entrypoint_name,
            trace_only_inside_code_string=True,
            max_valid_events=self._cfg.max_valid_events,
            max_events_per_line=self._cfg.max_events_per_line,
            max_var_repr_length=self._cfg.max_var_repr_length,
            use_safe_execution=False,
            seed=self._cfg.seed + hash((problem.problem_id.problem_idx, s_idx, t_idx, augment)) % 10000,
        )
        return trace_res, key

    def _gen_code(self, problem: traces_utils.CodingProblem, s_idx: int) -> str:
        # variants to make traces slightly different per solution index
        if self._cfg.code_kind == "script" or self._cfg.entrypoint_name is None:
            # use stdin-like mode in tracer (no entrypoint), but we'll still keep a function pattern
            # for code_blocks richness; however, tracing will run whole script body.
            return "total = 0\n" "for i in range(3):\n" "    total += i\n" "print(total)\n"
        else:
            # function-based code with deterministic operations and small variation per solution;
            # expect one input value `x` passed to the entrypoint.
            if s_idx % 2 == 0:
                return (
                    "def solution(x):\n"
                    "    x = int(x)\n"
                    "    total = 0\n"
                    "    for i in range(x):\n"
                    "        total += i\n"
                    "    print(total)\n"
                    "    return 2 * x\n"
                )
            else:
                return (
                    "def solution(x):\n"
                    "    x = int(x)\n"
                    "    val = x * x\n"
                    "    if x % 2 == 0:\n"
                    "        val = val + x\n"
                    "    else:\n"
                    "        val = val - x\n"
                    "    print(val)\n"
                    "    return 2 * x\n"
                )


class FakeDeltaDatasetReader(FakeTraceDatasetReader):
    """Fake replacement for pyine.data.deltas.dataset_reader.DatasetReader.

    Inherits fake traces generation and adds deltas for each trace. __getitem__ returns
    TraceDeltaList instead of TraceResult. Also supports indexing by the ".../deltas" key.
    """

    def __init__(
        self,
        lmdb_path: typing.Any | None = None,
        *,
        config: FakeTraceDataConfig | None = None,
        **kwargs,
    ) -> None:
        super().__init__(lmdb_path=lmdb_path, config=config, **kwargs)
        # build deltas indices/keys and precompute deltas
        self.deltas_indices: list[int] = []
        self.deltas_keys: list[str] = []
        self._deltas: list[deltas_utils.TraceDeltaList] = []
        for i, tkey in enumerate(self.trace_keys):
            dkey = f"{tkey}{traces_utils.DELTAS_SUFFIX}"
            self.deltas_keys.append(dkey)
            self.deltas_indices.append(i)
            dlist = self._make_deltas(self._traces[i])
            self._deltas.append(dlist)
        assert len(self.deltas_indices) == len(self._deltas) == len(self)

    def __getitem__(self, index_or_key: int | str) -> deltas_utils.TraceDeltaList:
        if isinstance(index_or_key, int):
            if not (0 <= index_or_key < len(self)):
                raise IndexError(f"index {index_or_key} out of range")
            idx = index_or_key
        elif isinstance(index_or_key, str):
            if index_or_key in self.deltas_keys:
                idx = self.deltas_keys.index(index_or_key)
            elif index_or_key in self.trace_keys:
                idx = self.trace_keys.index(index_or_key)
            else:
                # mirror real deltas reader behavior which asserts on invalid keys
                raise AssertionError(f"key {index_or_key} not found in fake deltas dataset")
        else:  # pragma: no cover
            raise ValueError(f"invalid index_or_key type: {type(index_or_key)}")
        return self._deltas[idx]

    def get_trace_data(self, index_or_key: int | str) -> exec_utils.TraceResult:
        # mirror real deltas reader convenience method
        return FakeTraceDatasetReader.__getitem__(self, index_or_key)

    # ---------------------------- internals ----------------------------

    def _make_deltas(self, trace_res: exec_utils.TraceResult) -> deltas_utils.TraceDeltaList:
        dlist = deltas_utils.get_deltas_from_trace_steps(
            trace_res=trace_res,
            delta_generator=deltas_utils.DeltaGeneratorType.SIMPLE,
            include_global_vars=True,
            verbose=False,
        )
        # ensure the trace_id is a string so it mirrors real datasets
        return deltas_utils.TraceDeltaList.model_validate(dlist.model_dump())
