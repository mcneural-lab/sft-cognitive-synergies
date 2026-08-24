"""Switch prediction methods for semantic fluency lists.

This code was modified from the forager package by the Lexicon Lab:
https://github.com/thelexiconlab/forager
"""

import numpy as np
import pandas as pd
from scipy import stats


def switch_median(fluency_list, semantic_similarity, z_score=False):
    """
    Predicts switches in a fluency list based on the median of similarity scores.

    This function identifies a switch if the similarity between an item and the
    preceding item falls below the median of all similarity scores for the list.
    It includes an option to z-score the similarities before comparison.

    Args:
        fluency_list (list): The fluency list to predict switches on.
        semantic_similarity (list): A list of RAW semantic similarities where
                                    semantic_similarity[k] is the similarity between
                                    fluency_list[k] and fluency_list[k-1].
        z_score (bool, optional): If True, z-scores the similarities before
                                  calculating the median and comparing.
                                  Defaults to False.

    Returns:
        list: A list of predicted switches, where 0 = no switch and 1 = switch.
    """
    # Replace NaN values in semantic_similarity with None
    semantic_similarity = [None if pd.isna(s) else s for s in semantic_similarity]

    # 1. Extract valid raw similarity scores to work with
    valid_raw_similarities = [s for s in semantic_similarity if s is not None]
    if not valid_raw_similarities:
        return [0] * len(fluency_list)  # Return all zeros if no similarities exist

    # 2. Optionally z-score the similarities
    if z_score:
        mean_sim = np.mean(valid_raw_similarities)
        std_sim = np.std(valid_raw_similarities)
        # Handle the edge case where all similarities are the same
        if std_sim == 0:
            scores_to_compare = [0 if s is not None else None for s in semantic_similarity]
        else:
            scores_to_compare = [(s - mean_sim) / std_sim if s is not None else None for s in semantic_similarity]
    else:
        scores_to_compare = semantic_similarity

    # 3. Calculate the median threshold from the chosen scores (raw or z-scored)
    valid_comparison_scores = [s for s in scores_to_compare if s is not None]
    threshold = np.median(valid_comparison_scores)

    # 4. Apply the calculated threshold to determine switches
    switches = []
    for k in range(len(fluency_list)):
        # The first item cannot be a switch
        if k == 0:
            switches.append(0)
            continue

        current_score = scores_to_compare[k]
        if current_score is not None and current_score < threshold:
            switches.append(1)
        else:
            switches.append(0)

    return switches


def switch_delta(fluency_list, semantic_similarity, rise_thresh, fall_thresh):
    """
    Delta Similarity Switch Method proposed by Nancy Lundin & Peter Todd.

    Args:
        fluency_list (list, size = L): fluency list to predict switches on
        semantic_similarity (list, size = L): a list of semantic similarities between items in the fluency list, obtained via create_history_variables
        rise_thresh (float): after a switch occurs, the threshold that the increase in z-scored similarity must exceed to be a cluster
        fall_thresh (float): while in a cluster, the threshold that the decrease in z-scored similarity must exceed to be a switch

    Returns:
        a list, size L, of switches, where 0 = no switch, 1 = switch, 2 = boundary case
    """
    if rise_thresh > 1 or rise_thresh < 0:
        raise Exception("Rise Threshold parameter must be within range [0,1]")

    if fall_thresh > 1 or fall_thresh < 0:
        raise Exception("Fall Threshold parameter must be within range [0,1]")

    switchVector = [2]  # first item designated with 2

    # obtain consecutive semantic similarities b/w responses
    # z-score similarities within participant
    similaritiesZ = stats.zscore(semantic_similarity[1:])
    medianSim = np.median(similaritiesZ)
    similaritiesZ = np.concatenate(([np.nan], similaritiesZ))

    # define subject level threshold = median (zscored similarities)
    firstSwitchSimThreshold = medianSim
    # for second item, if similarity < median, then switch, else cluster
    if similaritiesZ[1] < firstSwitchSimThreshold:
        switchVector.append(1)
    else:
        switchVector.append(0)

    currentState = switchVector[1]
    previousState = currentState

    # for all other items:
    for n in range(1, len(fluency_list) - 1):
        #   consider n-1, n, n+1 items

        simPrecedingToCurrentWord = similaritiesZ[n]

        simCurrentToNextWord = similaritiesZ[n + 1]
        if previousState == 0:  # if previous state was a cluster
            if fall_thresh < (
                simPrecedingToCurrentWord - simCurrentToNextWord
            ):  # similarity diff fell more than threshold
                currentState = 1  # switch
            else:
                currentState = 0  # cluster
        else:  # previous state was a switch
            if rise_thresh < (
                simCurrentToNextWord - simPrecedingToCurrentWord
            ):  # similarity diff is greater than our rise threshold
                currentState = 0  # cluster
            else:
                currentState = 1  # switch

        switchVector.append(currentState)
        previousState = currentState

    return switchVector


def switch_simdrop(fluency_list, semantic_similarity):
    """
    Similarity Drop Switch Method from Hills TT, Jones MN, Todd (2012).

    Args:
        fluency_list (list, size = L): fluency list to predict switches on
        semantic_similarity (list, size = L): a list of semantic similarities between items in the fluency list, obtained via create_history_variables

    Returns:
        a list, size L, of switches, where 0 = no switch, 1 = switch, 2 = boundary case
    """
    simdrop = []
    for k in range(len(fluency_list)):
        if k > 0 and k < (len(fluency_list) - 1):
            # simdrop
            if (semantic_similarity[k + 1] > semantic_similarity[k]) and (
                semantic_similarity[k - 1] > semantic_similarity[k]
            ):
                simdrop.append(1)

            else:
                simdrop.append(0)

        else:
            simdrop.append(2)

    return simdrop
