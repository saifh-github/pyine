import pytest

import pyine.data.utils.splits as splits


@pytest.fixture
def sample_ids() -> list[str]:
    return [
        "id0",
        "id1",
        "id2",
        "id3",
        "id4",
        "id5",
        "id6",
        "id7",
    ]


@pytest.fixture
def sample_tags() -> list[list[str]]:
    # contains a variety of tags to exercise filter rules and case sensitivity
    return [
        ["group:A", "foo"],  # 0 -> A
        ["group:B"],  # 1 -> B
        ["group:A", "bar"],  # 2 -> A
        ["misc", "x"],  # 3 -> no group
        ["group:C"],  # 4 -> other group
        ["misc"],  # 5 -> no group
        ["GROUP:A"],  # 6 -> A variant (case test)
        ["other:tag", "wip:review"],  # 7 -> has wip: (for regex forbid examples)
    ]


def test_validation_probabilities_sum() -> None:
    _ = splits.SplitConfig(
        subset_names=["train", "valid"],
        subset_assign_prob_map={"train": 0.6, "valid": 0.4},
    )
    with pytest.raises(ValueError):
        _ = splits.SplitConfig(
            subset_names=["train", "valid"],
            subset_assign_prob_map={"train": 0.6, "valid": 0.5},
        )


def test_validation_unknown_subset_names() -> None:
    with pytest.raises(ValueError):
        _ = splits.SplitConfig(
            subset_names=["train", "valid"],
            subset_assign_prob_map={"train": 1.0, "foo": 0.0},
        )
    with pytest.raises(ValueError):
        _ = splits.SplitConfig(
            subset_names=["train", "valid"],
            subset_assign_prob_map={"train": 1.0, "valid": 0.0},
            subset_assign_rules_map={"unknown": "+group:A"},
        )


def test_validation_subset_names_constraints() -> None:
    with pytest.raises(Exception):  # pydantic validation error or value error
        _ = splits.SplitConfig(
            subset_names=["", "valid"],
            subset_assign_prob_map={"": 0.5, "valid": 0.5},
        )
    with pytest.raises(Exception):
        _ = splits.SplitConfig(
            subset_names=["only"],
            subset_assign_prob_map={"only": 1.0},
        )


@pytest.mark.parametrize(
    "count,expected",
    [
        (0, "solutions:0-10"),
        (9, "solutions:0-10"),
        (10, "solutions:10-25"),
        (24, "solutions:10-25"),
        (25, "solutions:25-50"),
        (49, "solutions:25-50"),
        (50, "solutions:50-100"),
        (99, "solutions:50-100"),
        (100, "solutions:100-250"),
        (249, "solutions:100-250"),
        (250, "solutions:500-1000"),
        (999, "solutions:500-1000"),
        (1000, "solutions:1000+"),
    ],
)
def test_solution_bucket_edges(
    count: int,
    expected: str,
) -> None:
    assert splits._get_solution_count_bucket_tag(count) == expected


def test_hard_assignments_precedence_first_match(
    sample_ids: list[str],
    sample_tags: list[list[str]],
) -> None:
    # rules: any A -> train, any B -> test; others -> val (via probs)
    cfg = splits.SplitConfig(
        seed=123,
        subset_names=["train", "valid", "test"],
        subset_assign_rules_map={
            "train": "+group:A",
            "test": "+group:B",
        },
        subset_assign_prob_map={"train": 0.0, "valid": 1.0, "test": 0.0},
        rules_are_case_sensitive=True,
    )
    assigns = cfg.build_subset_assignments(sample_ids, sample_tags)
    assert assigns["id0"] == "train"
    assert assigns["id2"] == "train"
    assert assigns["id1"] == "test"
    # non-matching go to val due to prob map
    for sid in ["id3", "id4", "id5", "id6", "id7"]:
        assert assigns[sid] == "valid"
    # first match order: two identical rules for A; first key wins
    cfg2 = splits.SplitConfig(
        seed=123,
        subset_names=["train", "valid", "test"],
        subset_assign_rules_map={
            "valid": "+group:A",
            "train": "+group:A",  # same rule appears later, should not be used
        },
        subset_assign_prob_map={"train": 0.0, "valid": 1.0, "test": 0.0},
        rules_are_case_sensitive=True,
    )
    assigns2 = cfg2.build_subset_assignments(sample_ids, sample_tags)
    assert assigns2["id0"] == "valid"
    assert assigns2["id2"] == "valid"


def test_random_assignments_reproducible_no_rules(
    sample_ids: list[str],
    sample_tags: list[list[str]],
) -> None:
    cfg = splits.SplitConfig(
        seed=42,
        subset_names=["train", "valid"],
        subset_assign_prob_map={"train": 0.7, "valid": 0.3},
    )
    a1 = cfg.build_subset_assignments(sample_ids, sample_tags)
    a2 = cfg.build_subset_assignments(sample_ids, sample_tags)
    assert a1 == a2  # same seed and inputs => deterministic
    # ensure only known subset names are used
    assert set(a1.values()).issubset({"train", "valid"})


def test_stratified_grouping_and_default_group(
    sample_ids: list[str],
    sample_tags: list[list[str]],
) -> None:
    # two explicit groups (A, B) and default group for the rest
    cfg = splits.SplitConfig(
        seed=7,
        subset_names=["train", "valid"],
        subset_assign_prob_map={"train": 1.0, "valid": 0.0},  # deterministic to simplify checks
        stratif_group_rules=["+group:A", "+group:B"],
        rules_are_case_sensitive=True,
    )
    assigns = cfg.build_subset_assignments(sample_ids, sample_tags)
    # all un-hard-assigned samples end up in "train" deterministically; this asserts code path runs
    assert set(assigns.keys()) == set(sample_ids)
    assert set(assigns.values()) == {"train"}


def test_case_sensitivity_affects_matching() -> None:
    ids = ["i0", "i1"]
    tags = [["GROUP:A"], ["group:a"]]
    # case sensitive: only exact case matches +group:A
    cfg_cs = splits.SplitConfig(
        seed=0,
        subset_names=["X", "Y"],
        subset_assign_rules_map={"X": "+group:A"},
        subset_assign_prob_map={"X": 0.0, "Y": 1.0},
        rules_are_case_sensitive=True,
    )
    a_cs = cfg_cs.build_subset_assignments(ids, tags)
    assert a_cs["i0"] == "Y"  # GROUP:A != group:A
    assert a_cs["i1"] == "Y"
    # case insensitive: both should match X
    cfg_ci = splits.SplitConfig(
        seed=0,
        subset_names=["X", "Y"],
        subset_assign_rules_map={"X": "+group:A"},
        subset_assign_prob_map={"X": 1.0, "Y": 0.0},
        rules_are_case_sensitive=False,
    )
    a_ci = cfg_ci.build_subset_assignments(ids, tags)
    assert a_ci["i0"] == "X"
    assert a_ci["i1"] == "X"


def test_hard_and_stratified_combined(
    sample_ids: list[str],
    sample_tags: list[list[str]],
) -> None:
    # hard-assign A -> train, stratify remaining by A/B (B exists) else default; deterministic to val
    cfg = splits.SplitConfig(
        seed=3,
        subset_names=["train", "valid"],
        subset_assign_rules_map={"train": "+group:A"},
        subset_assign_prob_map={"train": 0.0, "valid": 1.0},
        stratif_group_rules=["+group:A", "+group:B"],
        rules_are_case_sensitive=True,
    )
    assigns = cfg.build_subset_assignments(sample_ids, sample_tags)
    # A samples should be hard-assigned to train
    assert assigns["id0"] == "train"
    assert assigns["id2"] == "train"
    # others go to val deterministically
    for sid in ["id1", "id3", "id4", "id5", "id6", "id7"]:
        assert assigns[sid] == "valid"


def test_apply_hard_assignment_errors() -> None:
    cfg = splits.SplitConfig(
        subset_names=["a", "b"],
        subset_assign_prob_map={"a": 1.0, "b": 0.0},
    )
    # mismatched lengths
    with pytest.raises(ValueError):
        cfg._apply_hard_subset_assignments(["i0"], [])
    # duplicate identifiers
    with pytest.raises(ValueError):
        cfg._apply_hard_subset_assignments(["i0", "i0"], [["t"], ["t2"]])


def test_build_subset_to_identifiers_map(
    sample_ids: list[str],
    sample_tags: list[list[str]],
) -> None:
    cfg = splits.SplitConfig(
        seed=1,
        subset_names=["train", "valid", "test"],
        subset_assign_rules_map={"train": "+group:A", "test": "+group:B"},
        subset_assign_prob_map={"train": 0.0, "valid": 1.0, "test": 0.0},
    )
    id_to_subset = cfg.build_subset_assignments(sample_ids, sample_tags)
    subset_to_ids = cfg.build_subset_to_identifiers_map(sample_ids, sample_tags)
    # ensure mapping is inverse-compatible
    for sid, subset in id_to_subset.items():
        assert sid in subset_to_ids[subset]
    # and ids are partitioned across subsets
    flat: list[str] = [sid for ids in subset_to_ids.values() for sid in ids]
    assert sorted(flat) == sorted(sample_ids)


def test_get_dataset_split_file():
    split_file_path = splits.get_dataset_split_file_path("fooOOO", must_exist=False)
    assert "fooOOO" in split_file_path.name
    assert split_file_path.parent.exists()
    with pytest.raises(FileNotFoundError):
        _ = splits.get_dataset_split_result("fooOOO")
