"""Load and preprocess the dyadic collaborative-experiment data.

Notes:
  * helper functions imported from `sftbench.dyadic.utils`
  * `data_dir` defaults to the frozen public-release bundle containing embeddings from the pre-processed word list under
    `data/dyadic/conceptnet`
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sftbench import find_project_root

from .utils import (
    all_nans_to_nan,
    assign_players,
    calculate_other_similarities,
    calculate_self_similarities,
    get_partner_value,
    normalize_embedding,
)


def default_data_dir() -> Path:
    """Path to the frozen dyadic data bundle under the project root."""
    root = find_project_root() or Path.cwd()
    return root / "data" / "dyadic" / "conceptnet"


# %%
def load_data(data_dir=None, cross_condition_only=False, include_solo=False, include_supermarket=False):
    # Load data
    if data_dir is None:
        data_dir = default_data_dir()
    data_dir = Path(data_dir)
    print("Loading data...")
    h_df = pd.read_csv(f"{data_dir}/solo_words_processed.csv")
    ha_df = pd.read_csv(f"{data_dir}/ai_words_processed.csv")
    hh_df = pd.read_csv(f"{data_dir}/hh_words_processed.csv")

    # load embeddings and similarity-calculated switches
    h_embed = pd.read_pickle(f"{data_dir}/solo_words_with_embeddings_switches.pkl")
    ha_embed = pd.read_pickle(f"{data_dir}/ai_words_with_embeddings_switches.pkl")
    hh_embed = pd.read_pickle(f"{data_dir}/hh_words_with_embeddings_switches.pkl")

    # Filter out "supermarket" category from ALL dataframes for consistency
    # (pass include_supermarket=True to retain it)
    excluded_categories = [] if include_supermarket else ["supermarket"]
    if excluded_categories:
        h_df = h_df[~h_df["category"].isin(excluded_categories)].reset_index(drop=True)
        ha_df = ha_df[~ha_df["category"].isin(excluded_categories)].reset_index(drop=True)
        hh_df = hh_df[~hh_df["category"].isin(excluded_categories)].reset_index(drop=True)
        h_embed = h_embed[~h_embed["category"].isin(excluded_categories)].reset_index(drop=True)
        ha_embed = ha_embed[~ha_embed["category"].isin(excluded_categories)].reset_index(drop=True)
        hh_embed = hh_embed[~hh_embed["category"].isin(excluded_categories)].reset_index(drop=True)
    print(f"Excluded categories: {excluded_categories}. Remaining categories: {ha_df['category'].unique()}")

    ha_embed["dyadType"] = "ha"
    hh_embed["dyadType"] = "hh"

    ha_embed["embedding"] = ha_embed["embedding"].apply(normalize_embedding)
    hh_embed["embedding"] = hh_embed["embedding"].apply(normalize_embedding)
    h_embed["embedding"] = h_embed["embedding"].apply(normalize_embedding)

    # get_partner_value preserves original indices, so we can assign directly without .values
    hh_embed["partner_embedding"] = hh_embed.groupby("gameID", group_keys=False).apply(
        lambda g: get_partner_value(g, "embedding"), include_groups=False
    )
    ha_embed["partner_embedding"] = ha_embed.groupby("playerID", group_keys=False).apply(
        lambda g: get_partner_value(g, "embedding"), include_groups=False
    )

    ha_embed["embedding_similarity"] = ha_embed.apply(
        lambda row: np.dot(row["embedding"], row["partner_embedding"])
        if isinstance(row["partner_embedding"], np.ndarray)
        else np.nan,
        axis=1,
    )
    hh_embed["embedding_similarity"] = hh_embed.apply(
        lambda row: np.dot(row["embedding"], row["partner_embedding"])
        if isinstance(row["partner_embedding"], np.ndarray)
        else np.nan,
        axis=1,
    )

    # for cases where all embedding dims are nan, set similarity to nan
    ha_embed["embedding_similarity"] = ha_embed["embedding_similarity"].apply(all_nans_to_nan)
    hh_embed["embedding_similarity"] = hh_embed["embedding_similarity"].apply(all_nans_to_nan)

    # For solo data, compute self-similarity (similarity with own previous word)
    h_embed["partner_embedding"] = h_embed.groupby(["playerID", "category"])["embedding"].shift(1)
    h_embed["embedding_similarity"] = h_embed.apply(
        lambda row: np.dot(row["embedding"], row["partner_embedding"]), axis=1
    )
    h_embed["embedding_similarity"] = h_embed["embedding_similarity"].apply(all_nans_to_nan)

    # merge dataframes
    h_df["dyadType"] = "solo"
    ha_df["dyadType"] = "ha"
    hh_df["dyadType"] = "hh"
    h_df["source"] = h_df.playerID
    h_df["sourceType"] = "human"
    hh_df["sourceType"] = "human"

    # Fixed player assignment: Player 1 = source that produced word_index 0 in each group
    # For HH dyads: group by gameID, for HA dyads: group by playerID
    hh_df = hh_df.groupby("gameID", group_keys=False)[hh_df.columns.tolist()].apply(assign_players)
    ha_df = ha_df.groupby("playerID", group_keys=False)[ha_df.columns.tolist()].apply(assign_players)

    ha_df.loc[ha_df["source"] == "user", "sourceType"] = "human"
    ha_df.loc[ha_df["source"] == "ai", "sourceType"] = "ai"
    ha_df.loc[ha_df["sourceType"] == "human", "source"] = ha_df["playerID"][ha_df["sourceType"] == "human"]

    # Add a column for the partner_irt from the last partner turn in the hh_df and ha_df datasets
    hh_df["partner_irt"] = hh_df.groupby("gameID", group_keys=False).apply(
        lambda g: get_partner_value(g, "irt"), include_groups=False
    )
    ha_df["partner_irt"] = ha_df.groupby("playerID", group_keys=False).apply(
        lambda g: get_partner_value(g, "irt"), include_groups=False
    )
    # For solo data, "partner_irt" is actually the previous word's irt from the same player
    h_df["partner_irt"] = h_df.groupby(["playerID", "category"])["irt"].shift(1)

    # Sort dataframes consistently before concatenation to ensure proper row alignment
    # Sort by playerID, gameID, and word_index to ensure consistent ordering
    sort_columns = ["playerID", "gameID", "category", "word_index"]
    ha_df = ha_df.sort_values(sort_columns).reset_index(drop=True)
    ha_embed = ha_embed.sort_values(sort_columns).reset_index(drop=True)
    hh_df = hh_df.sort_values(sort_columns).reset_index(drop=True)
    hh_embed = hh_embed.sort_values(sort_columns).reset_index(drop=True)
    h_df = h_df.sort_values(sort_columns).reset_index(drop=True)
    h_embed = h_embed.sort_values(sort_columns).reset_index(drop=True)

    # Drop overlapping columns from CSV dataframes before concatenation
    cols_to_add = ["embedding", "embedding_similarity", "switch_sim", "partner_embedding"]
    ha_df = ha_df.drop(columns=[col for col in cols_to_add if col in ha_df.columns])
    hh_df = hh_df.drop(columns=[col for col in cols_to_add if col in hh_df.columns])
    h_df = h_df.drop(columns=[col for col in cols_to_add if col in h_df.columns])

    # Verify alignment before concatenation.
    # Compare as strings to be robust against dtype differences between CSV (may produce
    # nullable Int64/Float64 in pandas ≥ 3.0) and pickle (stores legacy float64 with NaN).
    def _aligned(df1, df2, cols):
        s1 = df1[cols].astype(str).apply(tuple, axis=1)
        s2 = df2[cols].astype(str).apply(tuple, axis=1)
        return len(s1) == len(s2) and (s1 == s2).all()

    assert _aligned(ha_df, ha_embed, sort_columns), "HA dataframes not aligned!"
    assert _aligned(hh_df, hh_embed, sort_columns), "HH dataframes not aligned!"
    assert _aligned(h_df, h_embed, sort_columns), "Solo dataframes not aligned!"

    # Concatenate dataframes
    ha_df = pd.concat([ha_df, ha_embed[cols_to_add]], axis=1)
    hh_df = pd.concat([hh_df, hh_embed[cols_to_add]], axis=1)
    h_df = pd.concat([h_df, h_embed[cols_to_add]], axis=1)
    if include_solo:
        df = pd.concat([h_df, ha_df, hh_df])  # including solo data
    else:
        df = pd.concat([ha_df, hh_df])  # not including solo data
    df.reset_index(inplace=True, drop=True)

    # ------------------------------------------------------------------
    # partner_midpoint_sim (Panel C)
    df["partner_midpoint_sim"] = np.nan

    dyadic_human_mask = (df["sourceType"] == "human") & (df["dyadType"] != "solo")
    dyadic_human_idx = df.index[dyadic_human_mask]

    _hs = df.loc[dyadic_human_idx].copy().sort_values(["source", "category", "word_index"])
    _hs["_prev_self_emb"] = _hs.groupby(["source", "category"])["embedding"].shift(1)

    def _partner_midpoint_sim(row):
        partner = row["partner_embedding"]
        prev = row["_prev_self_emb"]
        curr = row["embedding"]

        # Normalise to unit sphere before any geometry
        def _unit(v):
            n = np.linalg.norm(v)
            return v / n if n > 1e-10 else None

        curr_u = _unit(curr) if isinstance(curr, np.ndarray) else None
        prev_u = _unit(prev) if isinstance(prev, np.ndarray) else None
        part_u = _unit(partner) if isinstance(partner, np.ndarray) else None

        if part_u is not None and prev_u is not None and curr_u is not None:
            # Spherical midpoint: normalised chord midpoint = geodesic midpoint
            midpoint_dir = prev_u + curr_u
            norm_mid = np.linalg.norm(midpoint_dir)
            if norm_mid > 1e-10:
                # cosine to spherical midpoint
                return float(np.dot(part_u, midpoint_dir / norm_mid))
        return np.nan

    df.loc[_hs.index, "partner_midpoint_sim"] = _hs.apply(_partner_midpoint_sim, axis=1)

    # partner_embedding was only needed as an intermediate for embedding_similarity
    # (pre-merge) and partner_midpoint_sim (above); it is not consumed downstream.
    df = df.drop(columns=["partner_embedding"])

    # Calculate self and other similarities using new utils functions
    # self_similarity: similarity to own previous word (all conditions)
    # other_similarity: similarity to partner's previous word (hh/ha only, NaN for solo)
    df["self_similarity"] = calculate_self_similarities(df)
    df["other_similarity"] = calculate_other_similarities(df)

    # replace adjacent with convergent
    df["prompt"] = df["prompt"].replace({"adjacent": "convergent"})

    ##%% preprocess data
    df["log_irt"] = df["irt"].apply(lambda x: np.log(x) if x > 0 else np.nan)

    # Z-score within each player across all their data (all dyad types and categories)
    # This allows meaningful cross-condition comparisons
    df["log_irt_zscore"] = df.groupby(["source"])["log_irt"].transform(lambda x: (x - x.mean()) / x.std())

    # Create median splits for visualization
    # PlayerID identifies unique sequences of words in each dyadType
    df["word_index_split"] = df["word_index"] > df.groupby(["dyadType", "playerID", "category"])[
        "word_index"
    ].transform("median")

    if cross_condition_only:
        # Get cross-condition sources (all human players who did both HH and HA)
        hh_human_sources = set(df[(df["dyadType"] == "hh") & (df["sourceType"] == "human")]["source"].unique())
        ha_human_sources = set(df[(df["dyadType"] == "ha") & (df["sourceType"] == "human")]["source"].unique())
        cross_condition_sources = hh_human_sources.intersection(ha_human_sources)

        print(f"Cross-condition human sources: {len(cross_condition_sources)}")

        # Find gameIDs where ALL human sources across ALL conditions are cross-condition
        all_human_data = df[df["sourceType"] == "human"]
        valid_gameids = set()

        for game_id in all_human_data["gameID"].unique():
            # Get ALL human sources for this gameID across both HH and HA conditions
            game_sources = set(all_human_data[all_human_data["gameID"] == game_id]["source"].unique())
            if game_sources.issubset(cross_condition_sources):
                valid_gameids.add(game_id)

        print(f"GameIDs where ALL human sources are cross-condition: {len(valid_gameids)}")
        print(f"Original data shape: {df.shape}")
        print(f"Original HH games: {df[df['dyadType'] == 'hh']['gameID'].nunique()}")
        print(f"Original HA games: {df[df['dyadType'] == 'ha']['gameID'].nunique()}")

        # Filter the dataframe to keep only valid gameIDs, and preserve solo data if include_solo=True
        if include_solo:
            df = df[(df["gameID"].isin(valid_gameids)) | (df["dyadType"] == "solo")].copy()
        else:
            df = df[df["gameID"].isin(valid_gameids)].copy()

        print(f"Filtered data shape: {df.shape}")
        print(f"Filtered HH games: {df[df['dyadType'] == 'hh']['gameID'].nunique()}")
        print(f"Filtered HA games: {df[df['dyadType'] == 'ha']['gameID'].nunique()}")
        if include_solo:
            print(f"Solo games preserved: {df[df['dyadType'] == 'solo']['gameID'].nunique()}")

    return df
