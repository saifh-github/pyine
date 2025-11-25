import textwrap

import pyine.utils.code.complexity_metrics


def test_counts_for_simple_function() -> None:
    """Ensure basic line-count metrics are computed exactly for a toy snippet."""
    code = textwrap.dedent(
        """
        \"\"\"
        Module
        docs
        \"\"\"

        def greet(name: str) -> str:
            # inline comment
            return f\"hi {name}\"
        """
    ).strip()
    metrics = pyine.utils.code.complexity_metrics.get_complexity_metrics(code)
    assert metrics.loc == 8
    assert metrics.lloc == 3
    assert metrics.sloc == 2
    assert metrics.comments == 1
    assert metrics.multi == 4
    assert metrics.blank == 1


def test_complexity_grows_when_code_expands() -> None:
    """Complex metrics should reflect added decision branches and symbols."""
    base_code = textwrap.dedent(
        """
        def classify(value: int) -> str:
            if value > 0:
                return "pos"
            return "neg"
        """
    ).strip()
    extended_code = textwrap.dedent(
        """
        def classify(value: int) -> str:
            if value > 0:
                if value % 2 == 0:
                    return "pos-even"
                return "pos-odd"
            if value == 0:
                return "zero"
            if value < -10:
                return "neg-large"
            return "neg"
        """
    ).strip()
    base_metrics = pyine.utils.code.complexity_metrics.get_complexity_metrics(base_code)
    extended_metrics = pyine.utils.code.complexity_metrics.get_complexity_metrics(extended_code)
    assert extended_metrics.cyclomatic_complexity_sum > base_metrics.cyclomatic_complexity_sum
    assert extended_metrics.halstead_volume > base_metrics.halstead_volume
    assert extended_metrics.halstead_effort > base_metrics.halstead_effort
    assert extended_metrics.maintainability_index < base_metrics.maintainability_index
