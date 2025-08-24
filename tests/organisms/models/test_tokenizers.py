import transformers

from pyine.organisms.models.tokenizers import get_tokenizer


def test_get_tokenizer_basic_functionality():
    """Test basic functionality of get_tokenizer with default settings."""
    model_name = "gpt2"  # Using a small model for testing
    tokenizer = get_tokenizer(model_name)
    assert isinstance(tokenizer, transformers.PreTrainedTokenizerBase)
    assert tokenizer.name_or_path == model_name
