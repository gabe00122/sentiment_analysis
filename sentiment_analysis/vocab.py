from typing import NamedTuple


class VocabDescribe(NamedTuple):
    total_tokens: int
    # special tokens
    special_tokens: int = 1
    reverse_token: int = 0
