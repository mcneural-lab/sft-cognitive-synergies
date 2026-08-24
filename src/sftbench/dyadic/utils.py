"""Shared helpers for the dyadic analyses."""

import re

import numpy as np
import pandas as pd

# labels and color maps

LABEL_MAP = {
    "collab_human": "Human-Human",
    "collab_ai_inferred": "Human-AI (Inferred)",
    "collab_ai_convergent": "Human-AI (Convergent)",
    "collab_ai_divergent": "Human-AI (Divergent)",
    "collab_ai": "Human-AI",
    "solo": "Solo",  # Individual
    # word_count.py specific labels
    "hh_instructions": "Human-Human",
    "ha_inferred": "Human-AI\n(Inferred)",
    "ha_convergent": "Human-AI\n(Convergent)",
    "ha_divergent": "Human-AI\n(Divergent)",
    "solo_solo": "Nominal",
}

COLOR_MAP = {
    "collab_human": "#1f77b4",  # blue
    "collab_ai_inferred": "#ff7f0e",  # orange
    "collab_ai_convergent": "#ffbb78",  # light orange
    "collab_ai_divergent": "#810f0f",  # dark orange
    "collab_ai": "#ff7f0e",  # orange (collapsed AI)
    "solo": "#aec7e8",  # light blue
    # word_count.py specific colors
    "hh_instructions": "#1f77b4",  # blue
    "ha_convergent": "#ff7f0e",  # orange
    "ha_divergent": "#ff7f0e",  # orange
    "ha_inferred": "#ff7f0e",  # orange
    "solo_solo": "#1f77b4",  # blue
}

COLOR_MAP_COLLAPSED = {
    "solo": COLOR_MAP["solo"],
    "collab_human": COLOR_MAP["collab_human"],
    "collab_ai": COLOR_MAP["collab_ai"],
}

# Helper functions


def normalize_embedding(embedding):
    return embedding / np.linalg.norm(embedding)


def get_partner_value(group, value_col):
    """Get the other dyad partner's previous value in ``value_col``, handling non-alternating turns.

    For each word, looks backward to the most recent word from a *different*
    source (the partner) and returns that word's ``value_col``. Returns NaN when
    no prior partner word exists. Used for partner embeddings and IRTs.
    """
    partner_values = []
    group = group.sort_values("word_index")  # Sort but don't reset index
    original_index = group.index  # Capture index AFTER sorting so positions align

    for i in range(len(group)):
        current_source = group.iloc[i]["source"]
        found = False
        for j in range(i - 1, -1, -1):
            if group.iloc[j]["source"] != current_source:
                partner_values.append(group.iloc[j][value_col])
                found = True
                break
        if not found:
            partner_values.append(np.nan)

    return pd.Series(partner_values, index=original_index)


def all_nans_to_nan(x):
    """Convert arrays of all NaNs to scalar NaN."""
    if isinstance(x, (list, np.ndarray)) and np.all(np.isnan(x)):
        return np.nan
    return x


def _scrub_summary_timestamp(text: str) -> str:
    """Blank the wall-clock stamp statsmodels writes into a summary table.

    ``summary().as_text()`` fills a ``Date:``/``Time:`` cell with the moment the
    table was rendered, so two runs of an unchanged model produce two different
    files. The stamp carries no information about the fit, and it is the only
    thing separating those runs, so it is replaced with a fixed placeholder of
    the same width -- the surrounding columns hold real statistics and their
    alignment has to survive.
    """

    def _blank(match: re.Match[str]) -> str:
        return match.group("label") + "-" * len(match.group("value"))

    return re.sub(
        r"^(?P<label>\s*(?:Date|Time):\s+)(?P<value>\S.*?)(?=\s{2,}\S|\s*$)",
        _blank,
        text,
        flags=re.MULTILINE,
    )


def save_model_summary(model, filepath):
    """Write a model summary, minus the timestamp that would break reproducibility."""
    with open(filepath, "w") as f:
        f.write(_scrub_summary_timestamp(model.summary().as_text()))


def calculate_self_similarities(df):
    """
    Calculate self-similarities (similarity to own previous word) for all conditions.

    This function computes the cosine similarity between each word and the previous word
    from the same source. Handles all dyad types (solo, hh, ha).

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing word data with 'embedding' column and normalized embeddings.
        Must have columns: 'dyadType', 'source', 'category', 'word_index', 'embedding'

    Returns
    -------
    pd.Series
        Series of self-similarities aligned with input df. First word from each source
        in each category will have NaN.

    Notes
    -----
    - For solo: similarity to player's own previous word
    - For hh/ha: similarity to source's own previous word (not partner's)
    - Uses backward lookup to handle non-alternating turns correctly
    - Returns NaN for first word from each source in each category
    - Assumes embeddings are already normalized (uses np.dot as cosine similarity)
    """

    self_similarities = {}

    # Process each (source, category) group
    for _seq_key, group in df.groupby(["source", "category"]):  # separates each individual sequence
        # Keep original index but sort by word_index
        group = group.sort_values("word_index")
        original_indices = group.index.tolist()

        for i in range(len(group)):
            orig_idx = original_indices[i]
            if i == 0:
                # First word from this source in this category: no previous self word
                self_similarities[orig_idx] = np.nan
            else:
                # Calculate similarity to previous word from same source
                # Note: using np.dot because embeddings are already normalized
                curr_emb = group.iloc[i]["embedding"]
                prev_emb = group.iloc[i - 1]["embedding"]
                sim = np.dot(curr_emb, prev_emb)
                self_similarities[orig_idx] = sim

    # Create Series with original df index order
    result = pd.Series([self_similarities.get(idx, np.nan) for idx in df.index], index=df.index)

    return result


def calculate_other_similarities(df):
    """
    Calculate other-similarities (similarity to partner's previous word) for dyadic conditions.

    This function computes the cosine similarity between each word and the previous word
    from a different source (the partner).
    Returns NaN for solo condition.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing word data with 'embedding' column and normalized embeddings.
        Must have columns: 'dyadType', 'source', 'category', 'word_index', 'embedding'

    Returns
    -------
    pd.Series
        Series of other-similarities aligned with input df.
        - NaN for all solo condition rows
        - NaN for first word in each game/session (no previous partner word)
        - Valid similarity values for subsequent words in hh/ha conditions

    Notes
    -----
    - For solo: always NaN (no partner)
    - For hh: similarity to partner's previous word (grouped by gameID)
    - For ha: similarity to partner's previous word (grouped by playerID)
    - Uses backward lookup to handle non-alternating turns correctly
    - Assumes embeddings are already normalized (uses np.dot as cosine similarity)
    """

    other_similarities = {}

    # Handle solo data: no partner, so similarity is always NaN
    solo_mask = df["dyadType"] == "solo"
    for idx in df[solo_mask].index:
        other_similarities[idx] = np.nan

    # Handle hh data: group by gameID (contains both players)
    if "hh" in df["dyadType"].values:
        hh_data = df[df["dyadType"] == "hh"]
        for _game_id, game_group in hh_data.groupby("gameID"):
            # Sort by word_index to get temporal order
            game_group = game_group.sort_values("word_index")
            original_indices = game_group.index.tolist()

            for i in range(len(game_group)):
                orig_idx = original_indices[i]
                current_source = game_group.iloc[i]["source"]
                current_emb = game_group.iloc[i]["embedding"]

                # Look backwards for most recent word from different source (partner)
                found = False
                for j in range(i - 1, -1, -1):
                    if game_group.iloc[j]["source"] != current_source:
                        partner_emb = game_group.iloc[j]["embedding"]
                        # Note: using np.dot because embeddings are already normalized
                        sim = np.dot(current_emb, partner_emb)
                        other_similarities[orig_idx] = sim
                        found = True
                        break

                if not found:
                    # First word in game or no previous partner word
                    other_similarities[orig_idx] = np.nan

    # Handle ha data: group by playerID (contains human and AI)
    if "ha" in df["dyadType"].values:
        ha_data = df[df["dyadType"] == "ha"]
        for _player_id, player_group in ha_data.groupby("playerID"):
            # Sort by word_index to get temporal order
            player_group = player_group.sort_values("word_index")
            original_indices = player_group.index.tolist()

            for i in range(len(player_group)):
                orig_idx = original_indices[i]
                current_source = player_group.iloc[i]["source"]
                current_emb = player_group.iloc[i]["embedding"]

                # Look backwards for most recent word from different source (partner)
                found = False
                for j in range(i - 1, -1, -1):
                    if player_group.iloc[j]["source"] != current_source:
                        partner_emb = player_group.iloc[j]["embedding"]
                        # Note: using np.dot because embeddings are already normalized
                        sim = np.dot(current_emb, partner_emb)
                        other_similarities[orig_idx] = sim
                        found = True
                        break

                if not found:
                    # First word in session or no previous partner word
                    other_similarities[orig_idx] = np.nan

    # Create Series with original df index order
    result = pd.Series([other_similarities.get(idx, np.nan) for idx in df.index], index=df.index)

    return result


def assign_players(group):
    """Assign players based on the source that produced word_index 0 in each group."""
    first_word_row = group[group["word_index"] == 0]
    if len(first_word_row) > 0:
        first_source = first_word_row["source"].iloc[0]
        group.loc[group["source"] == first_source, "player"] = 1
        group.loc[group["source"] != first_source, "player"] = 2
    return group


# statistics reporting


def holm_corrected_pvalues(test_results):
    """Holm-Bonferroni corrected p-values for one annotator's set of comparisons.

    ``statannotations`` applies ``comparisons_correction`` when it draws the
    significance stars, but leaves ``StatResult.pvalue`` uncorrected. Exported
    statistics must therefore recompute the correction, otherwise the CSVs
    disagree with the figure they accompany.

    Returns a list aligned with the annotations that carry a p-value, in order.
    """
    from statsmodels.stats.multitest import multipletests

    raw = [a.data.pvalue for a in test_results if hasattr(a, "data") and hasattr(a.data, "pvalue")]
    if not raw:
        return []
    return list(multipletests(raw, method="holm")[1])


def significance_stars(pval):
    """Star notation for a (corrected) p-value."""
    if pval < 0.001:
        return "***"
    if pval < 0.01:
        return "**"
    if pval < 0.05:
        return "*"
    return "ns"


def format_pvalue(pval):
    """Format a p-value for CSV export."""
    return f"{pval:.6f}" if pval >= 0.001 else f"{pval:.2e}"
