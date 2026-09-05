"""Text preprocessing for bank transaction narrations.

This is the single source of truth for how a narration is cleaned. It is used
both at training time (train.py) and at inference time (predictor.py) so the
two can never drift apart.

The stop-word list and cleaning rules are ported directly from the original
research notebook (Mono-personal-finance-management.ipynb) so results stay
consistent, but the logic no longer depends on downloading NLTK at runtime.
"""

from __future__ import annotations

import re

# English stop words (a static snapshot of NLTK's english list) plus the
# domain-specific noise tokens that the notebook added. Kept inline so the app
# has no runtime dependency on nltk downloads.
_ENGLISH_STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "you're", "you've", "you'll", "you'd", "your", "yours", "yourself",
    "yourselves", "he", "him", "his", "himself", "she", "she's", "her",
    "hers", "herself", "it", "it's", "its", "itself", "they", "them",
    "their", "theirs", "themselves", "what", "which", "who", "whom", "this",
    "that", "that'll", "these", "those", "am", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "having", "do", "does",
    "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because",
    "as", "until", "while", "of", "at", "by", "for", "with", "about",
    "against", "between", "into", "through", "during", "before", "after",
    "above", "below", "to", "from", "up", "down", "in", "out", "on", "off",
    "over", "under", "again", "further", "then", "once", "here", "there",
    "when", "where", "why", "how", "all", "any", "both", "each", "few",
    "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "s", "t", "can", "will",
    "just", "don", "don't", "should", "should've", "now", "d", "ll", "m",
    "o", "re", "ve", "y", "ain", "aren", "aren't", "couldn", "couldn't",
    "didn", "didn't", "doesn", "doesn't", "hadn", "hadn't", "hasn", "hasn't",
    "haven", "haven't", "isn", "isn't", "ma", "mightn", "mightn't", "mustn",
    "mustn't", "needn", "needn't", "shan", "shan't", "shouldn", "shouldn't",
    "wasn", "wasn't", "weren", "weren't", "won", "won't", "wouldn", "wouldn't",
}

# Domain-specific noise tokens carried over from the notebook. These are mostly
# reference-code fragments and personal names that appeared in the sample data
# and add no categorisation signal.
_DOMAIN_STOPWORDS = {
    "xx", "xxxx", "via", "gw", "flw", "f", "onb", "pg", "nig", "i", "ac",
    "kla", "ik", "bw", "ly", "eg", "kd", "est", "enzy", "eomo", "epo",
    "erelesusi", "frozen", "err", "error", "ese", "ernest", "buhari", "love",
    "failure", "failed", "ezzy", "ezinna", "ezeugbor", "ezeokeke", "express",
    "expre", "exceeded", "fagbule",
}

STOP_WORDS = sorted(_ENGLISH_STOPWORDS | _DOMAIN_STOPWORDS)

# Precompiled regexes for speed.
_RE_DIGITS = re.compile(r"[0-9]")
_RE_NON_ALPHA = re.compile(r"[^a-z -]")

_STOP_SET = set(STOP_WORDS)


def clean_text(text: str) -> str:
    """Normalise a raw bank narration into model-ready tokens.

    Steps (matching the original notebook):
      1. lowercase
      2. replace digits with spaces
      3. drop any character that is not a-z, space, or hyphen
      4. remove stop words
    """
    if text is None:
        return ""
    text = str(text).lower()
    text = _RE_DIGITS.sub(" ", text)
    text = _RE_NON_ALPHA.sub(" ", text)
    tokens = [tok for tok in text.split() if tok not in _STOP_SET]
    return " ".join(tokens)
