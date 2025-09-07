import pathlib

import pytest

import pyine.data.deltas.dataset_reader
import pyine.data.deltas.dataset_utils
import pyine.data.deltas.dataset_writer
import pyine.data.traces.dataset_reader
import pyine.data.traces.dataset_utils
import pyine.data.traces.dataset_writer
import pyine.utils.code.execution
import pyine.utils.filesystem
import pyine.utils.portability
import tests.data.utils.env_checks


def _check_deltas_ok(deltas):
    """Utility function that checks that all provided deltas are valid."""
    # first delta should be parent calling -> entrypoint
    assert deltas[0].event_relationship == pyine.data.deltas.dataset_utils.EventRelationship.ENTRYPOINT
    # all deltas should happen inside the code string (no external calls), and be chained+ordered
    expected_file_name = pyine.utils.code.execution.EXEC_TRACE_FILE_NAME
    prev_line, prev_step_idx = 0, -1
    for delta_idx, delta in enumerate(deltas):
        if delta_idx == 0:
            assert delta.event_relationship == pyine.data.deltas.dataset_utils.EventRelationship.ENTRYPOINT
            assert delta.curr_trace_key.file == pyine.utils.code.execution.EXEC_PARENT_FILE_NAME
        else:
            assert delta.curr_trace_key.file == expected_file_name
        assert delta.curr_trace_key.line == prev_line
        if delta_idx < len(deltas) - 1:
            assert delta.next_trace_key.file == expected_file_name
        else:
            assert delta.next_trace_key.file == pyine.utils.code.execution.EXEC_PARENT_FILE_NAME
            assert delta.next_trace_key.line == 0
        assert prev_step_idx < delta.trace_step_idx
        prev_line, prev_step_idx = delta.next_trace_key.line, delta.trace_step_idx


def test_dataset_paths(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure *get_new_dataset_path* always returns a non-existing, unique path."""
    # monkey-patch the root path used by the utils module to the temp dir
    monkeypatch.setattr(pyine.utils.filesystem, "get_data_root_path", lambda: tmp_path)
    path_a = pyine.data.deltas.dataset_utils.get_new_dataset_path("TACO", "v01")
    path_b = pyine.data.deltas.dataset_utils.get_new_dataset_path("TACO", "v02")
    assert path_a != path_b
    assert not path_a.exists() and not path_b.exists()
    path_a.mkdir(parents=True)
    found_dataset_path = pyine.data.deltas.dataset_utils.get_latest_dataset_path("TACO")
    assert found_dataset_path == path_a
    assert found_dataset_path.exists()
    path_b.mkdir(parents=True)
    found_dataset_path = pyine.data.deltas.dataset_utils.get_latest_dataset_path("TACO")
    assert found_dataset_path == path_b


@pytest.mark.parametrize("input_val", [0, 1, 2])
def test_delta_generation_with_raised_exception(input_val: int):
    """Test delta generation from trace steps with raised exceptions."""
    example_snippet = """\
in_val = int(input("Enter a value: "))
if not in_val:
    print("OK")
else:
    raise ValueError(f"{in_val}")
print("all done")
"""
    input_args = f"{input_val}\n"
    trace_result = pyine.utils.code.execution.execute_and_trace_code(
        code_string=example_snippet,
        inputs=input_args,
        identifier="dummy",
        trace_only_inside_code_string=True,
    )
    assert trace_result.code_string == example_snippet
    assert trace_result.inputs == input_args
    deltas = pyine.data.deltas.dataset_utils.get_deltas_from_trace_steps(
        trace_res=trace_result,
        delta_generator=pyine.data.deltas.dataset_utils.DeltaGeneratorType.SIMPLE,
    )
    _check_deltas_ok(deltas)
    if input_val == 0:
        assert len(deltas) == 5
        assert deltas[-1].exception is None
        assert trace_result.stdout == "OK\nall done\n"
        assert deltas[-2].stdout == "OK\n"
        assert deltas[-1].stdout == "all done\n"
    else:
        assert len(deltas) == 4
        assert deltas[-1].exception is not None
        assert deltas[-1].exception.type == "ValueError"
        assert deltas[-1].exception.message == str(input_val)
        assert not trace_result.stdout


def test_delta_generation_with_generator_expr():
    """Test delta generation from trace steps with generator expressions."""
    example_snippet = """\
in_vals = []
while True:
    curr_val = float(input("Enter a value: "))
    if not curr_val:
        break
    in_vals.append(curr_val)
res = sum((in_vals[n] ** in_vals[n]) for n in range(len(in_vals)))
print(f"The result is: {int(res)}")
"""
    example_input_args = """\
5.0
3.0
2.0
0
"""
    trace_result = pyine.utils.code.execution.execute_and_trace_code(
        code_string=example_snippet,
        inputs=example_input_args,
        identifier="dummy",
        trace_only_inside_code_string=True,
    )
    assert trace_result.code_string == example_snippet
    assert trace_result.inputs == example_input_args
    deltas = pyine.data.deltas.dataset_utils.get_deltas_from_trace_steps(
        trace_res=trace_result,
        delta_generator=pyine.data.deltas.dataset_utils.DeltaGeneratorType.SIMPLE,
    )
    assert len(deltas) == 28
    _check_deltas_ok(deltas)
    assert trace_result.stdout == "The result is: 3156\n"


def test_delta_generation_with_exception_propagation():
    """Test delta generation from trace steps with exception propagation."""
    example_snippet = """\
def do_thing(area: float) -> float:
    '''do thingy'''
    raise ValueError("123")
    return area

def calculate_area(length: float, width: float) -> float:
    '''Returns the area of the rectangle specified via length and width.

    Specifically: returns area = length * width.
    '''
    area = length * width

    print(f"The area of the rectangle is: {area:.2f} square units")
    output = do_thing(area)
    print("okie")
    return output

length = float(input("Enter the length: "))
width = float(input("Enter the width: "))
# try:
#     calculate_area(length, width)
# except ValueError as e:
#     print(f"An error occurred: {e}")
calculate_area(length, width)
print("all done")
"""
    example_input_args = """\
5.0
3.0
"""
    trace_result = pyine.utils.code.execution.execute_and_trace_code(
        code_string=example_snippet,
        inputs=example_input_args,
        identifier="dummy",
        trace_only_inside_code_string=True,
    )
    assert trace_result.code_string == example_snippet
    assert trace_result.inputs == example_input_args
    assert trace_result.exception is not None
    assert trace_result.exception.type == "ValueError"
    assert trace_result.exception.message == "123"
    assert "The area of the rectangle is: 15.00 square units" in trace_result.stdout
    valid_traced_steps = [s for s in trace_result.traced_steps if s is not None]
    assert len(valid_traced_steps) == 18, "expected 18 steps, got: " + str(len(valid_traced_steps))
    assert valid_traced_steps[0].trace_key.line == 0  # first event should always start at dummy line
    deltas = pyine.data.deltas.dataset_utils.get_deltas_from_trace_steps(
        trace_res=trace_result,
        delta_generator=pyine.data.deltas.dataset_utils.DeltaGeneratorType.SIMPLE,
    )
    assert len(deltas) == 12
    _check_deltas_ok(deltas)
    # last delta should be raise to parent
    assert deltas[-1].event_relationship == pyine.data.deltas.dataset_utils.EventRelationship.RAISE
    assert deltas[-1].exception is not None
    found_raise = False
    for delta in deltas:
        if delta.exception is not None:
            assert delta.event_relationship == pyine.data.deltas.dataset_utils.EventRelationship.RAISE
            assert delta.curr_trace_key.line == 3
            found_raise = True
            break
    assert found_raise, "expected to find a raise event"


@pytest.mark.slow
@pytest.mark.timeout(120)  # 2 minutes should be plenty, otherwise tracing is failing for all snippets
@pytest.mark.skipif(
    tests.data.utils.env_checks.TACO_DATASET_MISSING,
    reason="TACO dataset is missing, cannot create mini deltas dataset",
)
def test_mini_taco_deltas_dataset(
    tmp_path: pathlib.Path,
) -> None:
    """Checks that a mini deltas dataset can be created and read successfully."""
    traces_dataset_path = tmp_path / "mini_taco_traces_dataset"
    assert not traces_dataset_path.exists()
    pyine.data.traces.dataset_writer.write_dataset_from_taco(
        output_dataset_path=traces_dataset_path,
        max_output_traces=10,
        max_solutions_per_problem=2,
        max_tests_per_solution=1,
        min_solution_dissimilarity=0.1,
        verbose=True,
    )
    assert traces_dataset_path.exists()
    output_dataset_path = tmp_path / "mini_taco_deltas_dataset"
    assert not output_dataset_path.exists()
    delta_gen = pyine.data.deltas.dataset_utils.DeltaGeneratorType.SIMPLE
    pyine.data.deltas.dataset_writer.write_dataset(
        traces_dataset_name_or_path=traces_dataset_path,
        output_dataset_path=output_dataset_path,
        delta_generator=delta_gen,
        verbose=True,
    )
    assert output_dataset_path.exists()
    delta_reader = pyine.data.deltas.dataset_reader.DatasetReader(output_dataset_path)
    assert delta_reader.get_metadata()["delta_generator"] == delta_gen.value
    deltas_list_count = len(delta_reader)  # noqa
    assert deltas_list_count >= 10
    for deltas in delta_reader:
        _check_deltas_ok(deltas)
    # deltas dataset should still be compatible with traces dataset reader
    trace_reader = pyine.data.traces.dataset_reader.DatasetReader(output_dataset_path)
    assert len(trace_reader) == deltas_list_count
    assert trace_reader.get_metadata() == delta_reader.get_metadata()
    assert trace_reader.get_size_on_disk() == delta_reader.get_size_on_disk()
    assert trace_reader.get_parent_dataset_name() == delta_reader.get_parent_dataset_name()
    assert trace_reader.get_hash() == delta_reader.get_hash()
    # but traces dataset should not be compatible with deltas dataset reader
    with pytest.raises(RuntimeError):
        _ = pyine.data.deltas.dataset_reader.DatasetReader(traces_dataset_path)
