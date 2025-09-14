import json

import pyine.organisms.models.utils.openai as openai_utils


def test_default_pred_grader_fine_tune_method_config_getter():
    config = openai_utils.PredGraderFineTuneMethodConfig().get_openai_config()
    assert isinstance(config, dict)
    # should be json-serializable with no errors raised
    _ = json.dumps(config)
