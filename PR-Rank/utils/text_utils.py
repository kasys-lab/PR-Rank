from collections import defaultdict
from string import punctuation

from spacy.lang.en.stop_words import STOP_WORDS

PUNCTUATION = set(list(punctuation))


def shorten_text(text: str, text_length_limit: int) -> str:
    if len(text) <= text_length_limit:
        return text
    else:
        return text[:text_length_limit]


def count_valid_tokens(tokens: list[str]) -> dict[str, int]:
    counter = defaultdict(int)
    for token in tokens:
        token = token.lower()
        if token not in STOP_WORDS and token not in PUNCTUATION and token.isalnum():
            counter[token] += 1
    return dict(counter)
