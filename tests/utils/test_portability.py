import numpy as np
import pytest

import pyine.utils.portability as portability


class TestEstimateTolerance:

    def test_integer_values(self):
        rtol, atol = portability.estimate_tolerance("123")
        assert atol == 0.0
        assert 1e-15 <= rtol <= 1e-5
        assert not np.isclose(122, 123, rtol, atol)
        assert np.isclose(122.9999, 123, rtol, atol)

    def test_simple_decimal(self):
        rtol, atol = portability.estimate_tolerance("1.0")
        assert atol == 0.0
        assert 1e-15 <= rtol <= 1e-5
        assert not np.isclose(1, 2, rtol, atol)
        assert np.isclose(0.99999, 1, rtol, atol)

    def test_decimal_with_precision(self):
        rtol, atol = portability.estimate_tolerance("3.14159")
        expected_atol = 0.5 * (10 ** (-5))  # 5 decimal places
        assert atol == expected_atol
        assert 1e-15 <= rtol <= 1e-5
        assert not np.isclose(3.1415, 3.14159, rtol, atol)
        assert np.isclose(3.141592654, 3.14159, rtol, atol)

    def test_small_decimal(self):
        rtol, atol = portability.estimate_tolerance("0.001")
        expected_atol = 0.5 * (10 ** (-3))  # 3 decimal places
        assert atol == expected_atol
        assert 1e-15 <= rtol <= 1e-5
        assert not np.isclose(0.0001, 0.001, rtol, atol)
        assert np.isclose(0.0012, 0.001, rtol, atol)

    def test_scientific_notation_small(self):
        rtol, atol = portability.estimate_tolerance("1.23e-6")
        assert rtol == 1e-5  # magnitude < 1e-6
        expected_atol = 0.5 * (10 ** (-8))  # 3 sig digits, exp -6
        assert atol == expected_atol
        assert not np.isclose(1.23e-7, 1.23e-6, rtol, atol)
        assert np.isclose(1.2345e-6, 1.23e-6, rtol, atol)

    def test_scientific_notation_large(self):
        rtol, atol = portability.estimate_tolerance("2.5e10")
        assert atol == 0.0
        assert rtol == 1e-5
        assert not np.isclose(2.5e9, 2.5e10, rtol, atol)
        assert np.isclose(2.49998e10, 2.5e10, rtol, atol)

    def test_negative_values(self):
        rtol, atol = portability.estimate_tolerance("-3.14")
        expected_atol = 0.5 * (10 ** (-2))
        assert atol == expected_atol
        assert 1e-15 <= rtol <= 1e-5

    def test_zero_value(self):
        rtol, atol = portability.estimate_tolerance("0")
        assert atol == 0.0
        assert rtol == 1e-9

    def test_zero_with_decimal(self):
        rtol, atol = portability.estimate_tolerance("0.0")
        assert atol == 0.0
        assert rtol == 1e-9

    def test_whitespace_handling(self):
        rtol, atol = portability.estimate_tolerance("  3.14  ")
        expected_atol = 0.5 * (10 ** (-2))
        assert atol == expected_atol
        assert 1e-15 <= rtol <= 1e-5

    def test_invalid_string_raises_error(self):
        with pytest.raises(ValueError):
            portability.estimate_tolerance("not_a_number")

    def test_tolerance_bounds_applied(self):
        rtol, atol = portability.estimate_tolerance("1.0000000000000001")
        assert rtol >= 1e-15
        assert rtol <= 1e-5
