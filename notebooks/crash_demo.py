# this demo script shows how we can get strange crashes when tracing TACO solutions;
# the crashes happen silently, meaning they are likely out-of-memory issues or segfaults
# (if you come across any crash that is NOT a out-of-memory or segfault, let me know!)

# TODO @@@@@@ problem fixed as of 2025-07-30, delete this file once we finish reviewing the changeset

import pyine.data.taco.dataset_utils as taco_dataset_utils
import pyine.data.traces.dataset_writer as dataset_writer
import pyine.utils.code.execution
import pyine.utils.reprod

if __name__ == "__main__":

    # for proper logging and dotenv setup
    pyine.utils.reprod.entrypoint_setup()

    # make sure we have a local copy of the REPACAKAGED TACO dataset locally
    source_dataset_path = taco_dataset_utils.get_latest_repackaged_dataset_path()
    # feel free to update the path above if you prefer storing the dataset in a non-default location
    # (should be `<project_root>/data/TACO/repackaged/` by default)
    assert source_dataset_path.exists()

    # the output dataset path is not really important, it will by default also be relative to data root
    # (we don't specify it, so it will be auto-filled)
    output_dataset_path = None

    dataset_writer.write_dataset_from_taco(
        source_dataset_path=source_dataset_path,
        output_dataset_path=output_dataset_path,
        # the settings below don't matter much (the crashes are quite frequent across traced solutions)
        max_output_traces=100,
        max_valid_solutions_per_problem=5,
        max_traces_per_solution=5,
        minimum_solution_dissimilarity=0.1,
        verbose=True,
    )

    print("all done!")  # you should not get here; the program will silently crash and exit
