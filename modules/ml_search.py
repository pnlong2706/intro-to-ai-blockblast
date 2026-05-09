"""ML-guided Best-first Search (Module E — runtime side).

Trains a classifier over (state, action) features and biases the search
priority by the model's ``P(good)`` estimate.
"""

import heapq
from itertools import count

import pandas as pd

from .ml_features import BASE_FEATURE_COLUMNS, extract_action_features
from .search import evaluate_node, transition_cost
from .state import Node


def prepare_training_data(df):
    """
    Split a dataset DataFrame into ``(X, y, feature_columns)``.

    ``block_name`` is one-hot encoded; the resulting column list is returned
    so prediction-time feature rows can be reindexed to match.
    """
    X_base = df[BASE_FEATURE_COLUMNS].copy()
    y = df["label"].astype(int).copy()

    X = pd.get_dummies(X_base, columns=["block_name"], dtype=int)
    feature_columns = X.columns.tolist()

    return X, y, feature_columns


def prepare_action_row_for_model(state, action, feature_columns):
    """Prepare a single (state, action) row aligned with ``feature_columns``."""
    feats = extract_action_features(state, action)
    if feats is None:
        return None

    row = pd.DataFrame([feats])[BASE_FEATURE_COLUMNS]
    row = pd.get_dummies(row, columns=["block_name"], dtype=int)
    row = row.reindex(columns=feature_columns, fill_value=0)
    return row


def rank_actions_with_model(state, model, feature_columns, top_k=None):
    """
    Score every valid action of ``state`` with ``model.predict_proba`` and
    return a list of ``(p_good, action)`` tuples sorted by ``p_good`` desc.
    """
    actions = state.get_valid_actions()
    scored_actions = []

    for action in actions:
        row = prepare_action_row_for_model(state, action, feature_columns)
        if row is None:
            continue

        p_good = model.predict_proba(row)[0][1]
        scored_actions.append((p_good, action))

    scored_actions.sort(key=lambda x: x[0], reverse=True)

    if top_k is not None:
        scored_actions = scored_actions[:top_k]

    return scored_actions


def best_first_search_ml(
    initial_state,
    model,
    feature_columns,
    max_expansions=5000,
    w1=4,
    w2=2,
    w3=6,
    K=8,
    remaining_weight=5,
    ml_beta=3.0,
    action_top_k=None,
):
    """
    Best-first Search with ML-guided priority.

        ``priority = evaluate_node(n) - ml_beta * P(good | action)``

    A higher ``p_good`` lowers the priority, pushing the node closer to the
    front of the min-heap. ``action_top_k`` optionally restricts expansion
    to the K model-ranked actions.
    """
    root = Node(initial_state)
    root.priority = evaluate_node(root, remaining_weight=remaining_weight)

    frontier = []
    counter = count()
    heapq.heappush(frontier, (root.priority, next(counter), root))

    best_seen_score = {initial_state.key(): initial_state.current_score}
    best_terminal = None
    best_partial = root
    expansions = 0

    while frontier and expansions < max_expansions:
        _, _, node = heapq.heappop(frontier)
        state = node.state
        key = state.key()

        if state.current_score < best_seen_score.get(key, -1):
            continue

        if state.current_score > best_partial.state.current_score:
            best_partial = node
        elif state.current_score == best_partial.state.current_score:
            if len(state.available_blocks) < len(best_partial.state.available_blocks):
                best_partial = node

        if state.is_terminal():
            if best_terminal is None or state.current_score > best_terminal.state.current_score:
                best_terminal = node
            continue

        expansions += 1

        ranked_actions = rank_actions_with_model(
            state,
            model=model,
            feature_columns=feature_columns,
            top_k=action_top_k,
        )

        for p_good, action in ranked_actions:
            child_state = state.place_block(*action)
            if child_state is None:
                continue

            child_key = child_state.key()
            if child_state.current_score <= best_seen_score.get(child_key, -1):
                continue

            step_cost = transition_cost(
                state, action, child_state,
                w1=w1, w2=w2, w3=w3, K=K,
            )
            child = Node(
                state=child_state,
                parent=node,
                action=action,
                path_cost=node.path_cost + step_cost,
            )

            base_eval = evaluate_node(child, remaining_weight=remaining_weight)
            child.priority = base_eval - ml_beta * p_good

            best_seen_score[child_key] = child_state.current_score
            heapq.heappush(frontier, (child.priority, next(counter), child))

    if best_terminal is not None:
        return best_terminal, expansions, True
    return best_partial, expansions, False
