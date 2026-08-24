from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sftbench import find_project_root

SPELLING_NORMALIZATION_CSV = (find_project_root() or Path.cwd()) / "data" / "embeddings" / "spelling_normalization.csv"


@lru_cache(maxsize=1)
def load_spelling_normalization() -> dict[str, str]:
    """Read the hand-curated spelling table shipped with the release.

    Participants typed free-form responses, so one concept arrives in many
    surface forms ("tshirt", "t-shirt", "T shirt"). Each row of
    ``data/embeddings/spelling_normalization.csv`` maps one observed spelling
    (with spaces already stripped, as :func:`apply_dict_mapping_to_df` receives
    them) onto a canonical surface form. Targets may be single tokens (e.g.
    ``guinea_pig``) or space-separated multi-word forms (e.g. ``bald eagle``); the
    latter are split and mean-pooled by
    :func:`~sftbench.embeddings.embeddings.text_to_embedding`. Responses absent from
    the table pass through unchanged and are matched by :func:`find_closest_words`.
    """
    table = pd.read_csv(SPELLING_NORMALIZATION_CSV)
    return dict(zip(table["response"], table["normalized"], strict=True))


def find_closest_words(original_words: list[str], embedding_words: list[str]) -> dict[str, str]:
    """
    Find the closest word in embedding_words for each word in original_words
    using TF-IDF vectorization and cosine similarity on character n-grams.

    The match is orthographic. No minimum similarity threshold is applied: every
    input word is mapped to its nearest neighbour in the embedding vocabulary.
    """
    if not original_words or not embedding_words:
        return {}

    # Create TF-IDF vectorizer with character n-grams (2-4 chars)
    # This captures subword patterns and handles typos/variations well
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",  # Character n-grams with word boundaries
        ngram_range=(2, 4),  # 2-4 character n-grams
        lowercase=True,
    )
    embedding_vectors = vectorizer.fit_transform(embedding_words)
    original_vectors = vectorizer.transform(original_words)
    similarity_matrix = cosine_similarity(original_vectors, embedding_vectors)
    best_matches_idx = np.argmax(similarity_matrix, axis=1)
    return {original_words[i]: embedding_words[best_matches_idx[i]] for i in range(len(original_words))}


def apply_dict_mapping_to_df(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """Normalise the spellings in ``column_name`` against the curated table."""
    mapping = load_spelling_normalization()
    df[column_name] = df[column_name].map(mapping).fillna(df[column_name])
    return df
