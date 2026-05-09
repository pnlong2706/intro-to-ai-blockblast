"""ML feature extraction and dataset generation (Module E — data side).

These are the helpers used by the Decision Tree / Random Forest pipeline
to build per-action ``(state, action) -> good/bad`` training samples. The
labelling step uses :func:`best_first_search` as a teacher signal.
"""

import random

import numpy as np
import pandas as pd

from .blocks import TETRIMINOS
from .board import bounding_box_area, count_empty_components
from .search import best_first_search
from .state import State


BASE_FEATURE_COLUMNS = [
    "filled_ratio",
    "holes_before",
    "holes_after",
    "delta_holes",
    "bbox_before",
    "bbox_after",
    "delta_area",
    "remaining_blocks_before",
    "remaining_blocks_after",
    "valid_actions_before",
    "valid_actions_after",
    "near_rows_before",
    "near_cols_before",
    "near_rows_after",
    "near_cols_after",
    "lines_cleared",
    "score_gain",
    "block_cells",
    "block_h",
    "block_w",
    "x",
    "y",
    "x_norm",
    "y_norm",
    "block_name",
]


def count_nearly_full_lines(board, threshold=7):
    """Count rows / columns that have at least ``threshold`` filled cells."""
    row_counts = board.sum(axis=1)
    col_counts = board.sum(axis=0)
    near_rows = int(np.sum(row_counts >= threshold))
    near_cols = int(np.sum(col_counts >= threshold))
    return near_rows, near_cols


def generate_random_reachable_state(steps_range=(2, 6), blocks_per_turn=4, board_size=8):
    """
    Build a plausible mid-game starting state by placing a few random valid
    blocks on an empty board, then drawing ``blocks_per_turn`` random blocks
    as the available list for the next turn.
    """
    board = np.zeros((board_size, board_size), dtype=int)
    all_blocks = list(TETRIMINOS.keys())

    num_steps = random.randint(*steps_range)

    for _ in range(num_steps):
        block_name = random.choice(all_blocks)

        temp_state = State(
            board=board,
            available_blocks=[block_name],
            current_score=0,
        )

        actions = temp_state.get_valid_actions()
        if not actions:
            break

        action = random.choice(actions)
        next_state = temp_state.place_block(*action)
        if next_state is None:
            continue

        board = np.copy(next_state.board)

    available_blocks = random.sample(all_blocks, blocks_per_turn)

    return State(
        board=board,
        available_blocks=available_blocks,
        current_score=0,
    )


def extract_action_features(state, action):
    """
    Return a feature dict describing ``(state, action)`` after the action is
    applied (post line clearing). Returns ``None`` for invalid actions.
    """
    block_name, x, y = action

    result = state.place_block(block_name, x, y, return_info=True)
    if result is None:
        return None

    child_state, info = result
    if child_state is None or info is None:
        return None

    block = TETRIMINOS[block_name]
    h, w = block.shape

    holes_before = count_empty_components(state.board)
    holes_after = count_empty_components(child_state.board)

    bbox_before = bounding_box_area(state.board)
    bbox_after = bounding_box_area(child_state.board)

    near_rows_before, near_cols_before = count_nearly_full_lines(state.board, threshold=7)
    near_rows_after, near_cols_after = count_nearly_full_lines(child_state.board, threshold=7)

    return {
        "filled_ratio": state.board.sum() / 64.0,
        "holes_before": holes_before,
        "holes_after": holes_after,
        "delta_holes": holes_after - holes_before,
        "bbox_before": bbox_before,
        "bbox_after": bbox_after,
        "delta_area": bbox_after - bbox_before,
        "remaining_blocks_before": len(state.available_blocks),
        "remaining_blocks_after": len(child_state.available_blocks),
        "valid_actions_before": len(state.get_valid_actions()),
        "valid_actions_after": len(child_state.get_valid_actions()),
        "near_rows_before": near_rows_before,
        "near_cols_before": near_cols_before,
        "near_rows_after": near_rows_after,
        "near_cols_after": near_cols_after,
        "lines_cleared": info["lines_cleared"],
        "score_gain": child_state.current_score - state.current_score,
        "block_cells": int(block.sum()),
        "block_h": h,
        "block_w": w,
        "x": x,
        "y": y,
        "x_norm": x / 7.0,
        "y_norm": y / 7.0,
        "block_name": block_name,
    }


def label_actions_for_state(
    state,
    teacher_budget=250,
    positive_ratio=0.25,
    max_actions_per_state=25,
    w1=4,
    w2=2,
    w3=6,
    K=8,
    remaining_weight=5,
):
    """
    Per-state labelling: for every (state, action) pair, run
    :func:`best_first_search` from the resulting child state as a teacher and
    use its final score as a quality signal. The top ``positive_ratio``
    fraction of actions for *this* state are labelled ``1``.

    Labels are only meaningful **within a state**, not globally.
    """
    actions = state.get_valid_actions()

    if len(actions) == 0:
        return []

    if max_actions_per_state is not None and len(actions) > max_actions_per_state:
        actions = random.sample(actions, max_actions_per_state)

    scored_samples = []

    for action in actions:
        feats = extract_action_features(state, action)
        if feats is None:
            continue

        child_state = state.place_block(*action)
        if child_state is None:
            continue

        result = best_first_search(
            child_state,
            max_expansions=teacher_budget,
            w1=w1,
            w2=w2,
            w3=w3,
            K=K,
            remaining_weight=remaining_weight,
        )

        best_node, expansions, found_terminal = result

        if best_node is None:
            teacher_final_score = child_state.current_score
        else:
            teacher_final_score = best_node.state.current_score

        sample = dict(feats)
        sample["teacher_final_score"] = teacher_final_score
        sample["teacher_found_terminal"] = int(found_terminal)
        sample["teacher_expansions"] = expansions
        scored_samples.append(sample)

    if len(scored_samples) == 0:
        return []

    scored_samples.sort(
        key=lambda d: (
            d["teacher_final_score"],
            d["lines_cleared"],
            d["score_gain"],
            -d["delta_holes"],
        ),
        reverse=True,
    )

    positive_count = max(1, int(np.ceil(len(scored_samples) * positive_ratio)))

    for idx, sample in enumerate(scored_samples):
        sample["label"] = 1 if idx < positive_count else 0
        sample["rank_in_state"] = idx + 1

    return scored_samples


def build_ml_dataset(
    n_states=40,
    teacher_budget=250,
    positive_ratio=0.25,
    max_actions_per_state=25,
    w1=4,
    w2=2,
    w3=6,
    K=8,
    remaining_weight=5,
    verbose=True,
):
    """Build a DataFrame of ``(state, action) -> label`` samples."""
    dataset = []

    for i in range(n_states):
        state = generate_random_reachable_state()

        samples = label_actions_for_state(
            state,
            teacher_budget=teacher_budget,
            positive_ratio=positive_ratio,
            max_actions_per_state=max_actions_per_state,
            w1=w1,
            w2=w2,
            w3=w3,
            K=K,
            remaining_weight=remaining_weight,
        )

        dataset.extend(samples)

        if verbose and (i + 1) % 10 == 0:
            print(f"Generated {i + 1}/{n_states} states | current samples = {len(dataset)}")

    return pd.DataFrame(dataset)
