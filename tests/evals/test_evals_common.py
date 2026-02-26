import pytest
import pytest_mock

import pyine.configs.schemas
import pyine.evals.common


class TestEvalType:
    def test_code_exec_enum(self) -> None:
        assert hasattr(pyine.evals.common.EvalType, "CODE_EXEC")
        assert pyine.evals.common.EvalType.CODE_EXEC == "code_exec"
        assert str(pyine.evals.common.EvalType.CODE_EXEC) == "code_exec"

    def test_correctness_enum(self) -> None:
        assert hasattr(pyine.evals.common.EvalType, "CORRECTNESS")
        assert pyine.evals.common.EvalType.CORRECTNESS == "correctness"
        assert str(pyine.evals.common.EvalType.CORRECTNESS) == "correctness"


class TestBaseEvalsConfig:
    @pytest.mark.asyncio
    async def test_evaluate_runnable_model_returns_empty_dict_when_eval_type_none(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        config = pyine.evals.common.BaseEvalsConfig()
        mock_chain = mocker.MagicMock()
        mock_datamodule = mocker.MagicMock()
        result = await config.evaluate_runnable_model(
            chain=mock_chain,
            datamodule=mock_datamodule,
            eval_subset_name="test",
        )
        assert result.metrics == {}

    @pytest.mark.asyncio
    async def test_evaluate_runnable_model_raises_when_eval_type_set(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        config = pyine.evals.common.BaseEvalsConfig(eval_type=pyine.evals.common.EvalType.CODE_EXEC)
        mock_chain = mocker.MagicMock()
        mock_datamodule = mocker.MagicMock()
        with pytest.raises(NotImplementedError, match="evaluation type code_exec not implemented"):
            await config.evaluate_runnable_model(
                chain=mock_chain,
                datamodule=mock_datamodule,
                eval_subset_name="test",
            )

    @pytest.mark.asyncio
    async def test_evaluate_hf_model_returns_empty_dict_when_eval_type_none(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        config = pyine.evals.common.BaseEvalsConfig()
        mock_model = mocker.MagicMock()
        mock_tokenizer = mocker.MagicMock()
        mock_datamodule = mocker.MagicMock()
        result = await config.evaluate_hf_model(
            model=mock_model,
            tokenizer=mock_tokenizer,
            datamodule=mock_datamodule,
            eval_subset_name="test",
        )
        assert result.metrics == {}

    @pytest.mark.asyncio
    async def test_evaluate_hf_model_raises_when_eval_type_set(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        config = pyine.evals.common.BaseEvalsConfig(eval_type=pyine.evals.common.EvalType.CODE_EXEC)
        mock_model = mocker.MagicMock()
        mock_tokenizer = mocker.MagicMock()
        mock_datamodule = mocker.MagicMock()
        with pytest.raises(NotImplementedError, match="evaluation type code_exec not implemented"):
            await config.evaluate_hf_model(
                model=mock_model,
                tokenizer=mock_tokenizer,
                datamodule=mock_datamodule,
                eval_subset_name="test",
            )

    def test_define_metrics_for_wandb_no_error_when_eval_type_none(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        config = pyine.evals.common.BaseEvalsConfig()
        mock_wandb_run = mocker.MagicMock()
        config.define_metrics_for_wandb(wandb_run=mock_wandb_run, eval_subset_names=["test"])

    def test_define_metrics_for_wandb_raises_when_eval_type_set(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        config = pyine.evals.common.BaseEvalsConfig(eval_type=pyine.evals.common.EvalType.CODE_EXEC)
        mock_wandb_run = mocker.MagicMock()
        with pytest.raises(NotImplementedError, match="evaluation type code_exec not implemented"):
            config.define_metrics_for_wandb(wandb_run=mock_wandb_run, eval_subset_names=["test"])

    def test_log_metrics_returns_none_when_eval_type_none(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        config = pyine.evals.common.BaseEvalsConfig()
        mock_wandb_run = mocker.MagicMock()
        result = config.log_metrics(
            wandb_run=mock_wandb_run,
            results_by_subset={"test": {}},
        )
        assert result is None

    def test_log_metrics_raises_when_eval_type_set(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        config = pyine.evals.common.BaseEvalsConfig(eval_type=pyine.evals.common.EvalType.CODE_EXEC)
        mock_wandb_run = mocker.MagicMock()
        with pytest.raises(NotImplementedError, match="evaluation type code_exec not implemented"):
            config.log_metrics(
                wandb_run=mock_wandb_run,
                results_by_subset={"test": {}},
            )

    def test_log_predictions_returns_none_when_eval_type_none(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        config = pyine.evals.common.BaseEvalsConfig()
        mock_wandb_run = mocker.MagicMock()
        result = config.log_predictions(
            wandb_run=mock_wandb_run,
            subset_name="test",
            subset_results={},
        )
        assert result is None

    def test_log_predictions_raises_when_eval_type_set(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        config = pyine.evals.common.BaseEvalsConfig(eval_type=pyine.evals.common.EvalType.CODE_EXEC)
        mock_wandb_run = mocker.MagicMock()
        with pytest.raises(NotImplementedError, match="evaluation type code_exec not implemented"):
            config.log_predictions(
                wandb_run=mock_wandb_run,
                subset_name="test",
                subset_results={},
            )

    @pytest.mark.asyncio
    async def test_evaluate_wrapped_model_returns_empty_dict_when_eval_type_none(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        config = pyine.evals.common.BaseEvalsConfig()
        result = await config.evaluate_wrapped_model(wrapped_model=[mocker.MagicMock()])
        assert result.metrics == {}

    @pytest.mark.asyncio
    async def test_evaluate_wrapped_model_accepts_single_model(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        config = pyine.evals.common.BaseEvalsConfig()
        result = await config.evaluate_wrapped_model(wrapped_model=mocker.MagicMock())
        assert result.metrics == {}

    @pytest.mark.asyncio
    async def test_evaluate_wrapped_model_raises_when_eval_type_set(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        config = pyine.evals.common.BaseEvalsConfig(eval_type=pyine.evals.common.EvalType.CODE_EXEC)
        with pytest.raises(NotImplementedError, match="evaluation type code_exec not implemented"):
            await config.evaluate_wrapped_model(wrapped_model=[mocker.MagicMock()])
