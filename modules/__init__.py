"""BlockBlast AI — module package.

Re-exports the most commonly used names so callers can write::

    from modules import State, Node, best_first_search, TETRIMINOS

Module map (mirrors the report chapters):

- :mod:`modules.blocks`      — Section 1 / Module A (data): Tetrimino shapes
- :mod:`modules.board`       — Section 2.1: board utilities
- :mod:`modules.state`       — Section 2.2: ``State`` and ``Node``
- :mod:`modules.search`      — Sections 2.3 + 3 + 4: cost, heuristic,
                               Best-first Search with the KR layer integrated
- :mod:`modules.kr_layer`    — Module C / Chapter 4 predicates
- :mod:`modules.ml_features` — Module E (data side): feature extraction +
                               teacher-signal labelling
- :mod:`modules.ml_search`   — Module E (runtime side): ``best_first_search_ml``
- :mod:`modules.bayes_risk`  — Module D / Chapter 5: rollout-based Bayes Network
"""

from .blocks import TETRIMINOS, base
from .board import (
    apply_block_and_clear,
    bounding_box_area,
    clear_full_lines,
    count_empty_components,
    get_empty_components,
)
from .kr_layer import (
    action_completes_line,
    compute_coverage_mask,
    has_trapped_cell,
)
from .ml_features import (
    BASE_FEATURE_COLUMNS,
    build_ml_dataset,
    count_nearly_full_lines,
    extract_action_features,
    generate_random_reachable_state,
    label_actions_for_state,
)
from .ml_search import (
    best_first_search_ml,
    prepare_action_row_for_model,
    prepare_training_data,
    rank_actions_with_model,
)
from .search import (
    best_first_search,
    evaluate_node,
    print_solution,
    reconstruct_node_chain,
    reconstruct_path,
    transition_cost,
)
from .state import Node, State

__all__ = [
    # blocks
    "TETRIMINOS",
    "base",
    # board
    "apply_block_and_clear",
    "bounding_box_area",
    "clear_full_lines",
    "count_empty_components",
    "get_empty_components",
    # state
    "Node",
    "State",
    # search
    "best_first_search",
    "evaluate_node",
    "print_solution",
    "reconstruct_node_chain",
    "reconstruct_path",
    "transition_cost",
    # kr_layer
    "action_completes_line",
    "compute_coverage_mask",
    "has_trapped_cell",
    # ml_features
    "BASE_FEATURE_COLUMNS",
    "build_ml_dataset",
    "count_nearly_full_lines",
    "extract_action_features",
    "generate_random_reachable_state",
    "label_actions_for_state",
    # ml_search
    "best_first_search_ml",
    "prepare_action_row_for_model",
    "prepare_training_data",
    "rank_actions_with_model",
]
