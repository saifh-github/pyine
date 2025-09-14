import pathlib

import click.testing
import pytest

import pyine.apps.annotate.trace_annot_generator as app
import pyine.prompts
import tests.data.utils.env_checks as env_checks


@pytest.mark.slow
@pytest.mark.skipif(
    env_checks.OPENAI_API_KEY_MISSING,
    reason="OPENAI key missing, cannot check annotator app",
)
@pytest.mark.skipif(
    env_checks.TACO_TRACES_DATASET_MISSING,
    reason="TACO traces dataset is missing, cannot check annotation generation",
)
def test_main_trace_annot_generator(
    tmp_path: pathlib.Path,
) -> None:
    """End-to-end exercise of main() using the TACO dataset and a max sample count of 10."""
    db_file_path = pathlib.Path(tmp_path) / "temp.db"
    assert not db_file_path.exists()
    cli_runner = click.testing.CliRunner()
    cli_args = [
        "--dataset-latest-from=TACO",
        "--prompt-name=code_summary",
        '--prompt-vars={"target_word_count":"100"}',
        "--llm-option=provider=openai",
        "--llm-option=model=gpt-4o-mini",
        "--target-indices=30-40",
        f"--db-path={db_file_path}",
    ]
    res = cli_runner.invoke(app.main, cli_args)  # noqa
    assert res.exit_code == 0, res
    assert db_file_path.exists()
    db = pyine.prompts.PromptResultDB(db_file_path)
    ids = db.list_identifiers()
    assert len(ids) > 0
