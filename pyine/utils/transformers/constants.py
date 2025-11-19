__all__ = [
    "default_ignore_index",
]

default_ignore_index: int = -100  # this is an extremely-commonly-used default in pytorch/huggingface
"""The default index to ignore when computing the loss on example tokens."""
