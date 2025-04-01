import src.utils.code_exec

exec_w_mocks = src.utils.code_exec.execute_code_with_mocked_input


def test_basic_input_mocking():
    """Test basic input mocking functionality."""
    code = \
"""\
name = input("What's your name? ")
age = input("What's your age? ")
print(f"Hello, {name}! You are {age} years old.")
"""
    inputs = "Alice\n30"
    output, error = exec_w_mocks(code, inputs)
    assert error is None
    assert "Hello, Alice! You are 30 years old." in output


def test_sys_stdin_readline():
    """Test mocking of sys.stdin.readline()."""
    code = \
"""\
import sys
print("Enter your name:")
name = sys.stdin.readline().strip()
print("Enter your country:")
country = sys.stdin.readline().strip()
print(f"{name} is from {country}.")
"""
    inputs = "Bob\nUSA"
    output, error = exec_w_mocks(code, inputs)
    assert error is None
    assert "Bob is from USA." in output


def test_sys_stdin_read():
    """Test mocking of sys.stdin.read()."""
    code = \
"""\
import sys
print("Enter multiple lines of text:")
text = sys.stdin.read()
word_count = len(text.split())
print(f"You entered {word_count} words.")
"""
    inputs = "This is a test.\nMultiple lines\nof text."
    output, error = exec_w_mocks(code, inputs)
    assert error is None
    assert "You entered 8 words." in output


def test_sys_stdin_readlines():
    """Test mocking of sys.stdin.readlines()."""
    code = \
"""\
import sys
print("Enter multiple lines:")
lines = sys.stdin.readlines()
print(f"You entered {len(lines)} lines.")
print(f"First line: {lines[0].strip()}")
"""
    inputs = "First line\nSecond line\nThird line"
    output, error = exec_w_mocks(code, inputs)
    assert error is None
    assert "You entered 3 lines." in output
    assert "First line: First line" in output


def test_mixed_input_methods():
    """Test mixing different input methods."""
    code = \
"""\
import sys
name = input("Enter name: ")
print("Enter address:")
address = sys.stdin.readline().strip()
print("Enter additional info:")
info = sys.stdin.read()
print(f"Name: {name}, Address: {address}, Info: {info.strip()}")
"""
    inputs = "Charlie\n123 Main St\nExtra info\nMore details"
    output, error = exec_w_mocks(code, inputs)
    assert error is None
    assert "Name: Charlie, Address: 123 Main St, Info: Extra info\nMore details" in output


def test_not_enough_inputs():
    """Test behavior when not enough inputs are provided."""
    code = \
"""\
name = input("What's your name? ")
age = input("What's your age? ")
country = input("What's your country? ")
"""
    inputs = "David\n42"
    output, error = exec_w_mocks(code, inputs)
    assert error is not None and isinstance(error, EOFError)


def test_error_in_executed_code():
    """Test behavior when the executed code contains an error."""
    code = \
"""\
x = 10
y = 0
result = x / y  # Division by zero error
"""
    inputs = ""
    output, error = exec_w_mocks(code, inputs)
    assert error is not None and isinstance(error, ZeroDivisionError)


def test_empty_input():
    """Test with empty input strings."""
    code = \
"""\
response = input("Press Enter to continue...")
print("You pressed Enter")
"""
    inputs = ""
    output, error = exec_w_mocks(code, inputs)
    assert error is not None  # should get EOF error with empty inputs

    inputs = "\n"  # now try with a single empty line
    output, error = exec_w_mocks(code, inputs)
    assert error is None and "You pressed Enter" in output
