"""Best-first Search with the KR layer integrated.

Module A + B + C of the report: cost / heuristic / search and the predicate
pruning + completion bonus from Chapter 4. Setting ``prune_trapped=False`` and
``completion_bonus=0.0`` reproduces the pure baseline behaviour.
"""

import heapq
from itertools import count

from .board import bounding_box_area, count_empty_components
from .kr_layer import action_completes_line, has_trapped_cell
from .state import Node


def transition_cost(prev_state, action, next_state, w1=4, w2=2, w3=6, K=8):
    """
    Weighted penalty function:

        Cost = w1 * dHoles + w2 * dArea - w3 * dLinesCleared + K
    """
    prev_holes = count_empty_components(prev_state.board)
    next_holes = count_empty_components(next_state.board)
    delta_holes = next_holes - prev_holes

    prev_area = bounding_box_area(prev_state.board)
    next_area = bounding_box_area(next_state.board)
    delta_area = next_area - prev_area

    _, info = prev_state.place_block(*action, return_info=True)
    delta_lines_cleared = info["lines_cleared"] if info is not None else 0

    cost = (
        w1 * delta_holes
        + w2 * delta_area
        - w3 * delta_lines_cleared
        + K
    )
    return max(0, cost)


def evaluate_node(node, remaining_weight=5):
    """``Eval(n) = g(n) + remaining_weight * |remaining_blocks(n)|``."""
    return node.path_cost + remaining_weight * len(node.state.available_blocks)


def reconstruct_path(node):
    """Action sequence from the root to ``node``."""
    actions = []
    current = node
    while current is not None and current.action is not None:
        actions.append(current.action)
        current = current.parent
    actions.reverse()
    return actions


def reconstruct_node_chain(node):
    """List of nodes from the root to ``node`` (inclusive)."""
    chain = []
    current = node
    while current is not None:
        chain.append(current)
        current = current.parent
    chain.reverse()
    return chain


def print_solution(result):
    """Pretty-print a search result tuple ``(node, expansions, found_terminal)``."""
    node, expansions, found_terminal = result

    if node is None:
        print("No solution found.")
        return

    actions = reconstruct_path(node)

    print("Expanded nodes:", expansions)
    print("Terminal found:", found_terminal)
    print("Best score:", node.state.current_score)
    print("Placed blocks:", len(actions))
    print("Remaining blocks:", node.state.available_blocks)

    print("\nAction sequence:")
    for step, action in enumerate(actions, start=1):
        print(f"  Step {step}: place {action[0]} at ({action[1]}, {action[2]})")

    print("\nFinal board:")
    print(node.state.board)


def best_first_search(
    initial_state,
    max_expansions=5000,
    w1=4,
    w2=2,
    w3=6,
    K=8,
    remaining_weight=5,
    prune_trapped=True,
    completion_bonus=10.0,
):
    """
    Best-first Search for one-turn block placement.

    Base evaluation:
        ``Eval(n) = g(n) + remaining_weight * |remaining_blocks(n)|``

    Two knowledge-layer integrations from Chapter 4 sit on top:

    1. ``prune_trapped``: skip children where any cell satisfies
       ``Trapped(i,j,s)`` on the post-clear board.
    2. ``completion_bonus``: subtract this constant from priority when an
       action satisfies ``Completes(b,x,y,s)``.

    Setting ``prune_trapped=False`` and ``completion_bonus=0.0`` reproduces
    the pure baseline (an ablation that disables the knowledge layer).

    Returns:
        ``(best_terminal or best_partial, expansions, found_terminal)``
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

        if node.state.current_score > best_partial.state.current_score:
            best_partial = node
        elif node.state.current_score == best_partial.state.current_score:
            if len(node.state.available_blocks) < len(best_partial.state.available_blocks):
                best_partial = node

        if state.is_terminal():
            if best_terminal is None or state.current_score > best_terminal.state.current_score:
                best_terminal = node
            continue

        expansions += 1

        for action in state.get_valid_actions():
            child_state = state.place_block(*action)
            if child_state is None:
                continue

            child_key = child_state.key()

            if child_state.current_score <= best_seen_score.get(child_key, -1):
                continue

            if prune_trapped and has_trapped_cell(child_state):
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
            if completion_bonus > 0 and action_completes_line(state, action):
                child.priority = base_eval - completion_bonus
            else:
                child.priority = base_eval

            best_seen_score[child_key] = child_state.current_score
            heapq.heappush(frontier, (child.priority, next(counter), child))

    if best_terminal is not None:
        return best_terminal, expansions, True
    return best_partial, expansions, False
