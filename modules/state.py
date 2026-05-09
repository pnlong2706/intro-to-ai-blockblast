"""``State`` and ``Node`` classes for one-turn search.

Section 2.2 of the notebook. ``State`` holds ``(board, available_blocks,
current_score)``; ``current_score`` accumulates the number of cleared cells
(not lines). ``State.key()`` is the hashable identity used for de-duplication.
``Node`` adds parent / action / path_cost / priority / depth for search.
"""

import numpy as np

from .blocks import TETRIMINOS
from .board import apply_block_and_clear


class State:
    def __init__(self, board, available_blocks, current_score=0):
        """
        State for one-turn search.

        Args:
            board: 8x8 numpy array
            available_blocks: list of remaining blocks in the current turn
            current_score: accumulated score in this turn (cleared cells)
        """
        self.board = np.copy(board)
        self.available_blocks = list(available_blocks)
        self.current_score = current_score

    def is_goal(self):
        return len(self.available_blocks) == 0

    def key(self):
        """Hashable state representation (used for de-duplication)."""
        return (
            tuple(self.board.flatten()),
            tuple(sorted(self.available_blocks)),
        )

    def can_place(self, block_name, x, y):
        if block_name not in TETRIMINOS:
            return False

        block = TETRIMINOS[block_name]
        h, w = block.shape
        H, W = self.board.shape

        if x < 0 or y < 0 or x + h > H or y + w > W:
            return False

        for i in range(h):
            for j in range(w):
                if block[i, j] == 1 and self.board[x + i, y + j] == 1:
                    return False

        return True

    def get_valid_actions(self):
        actions = []
        H, W = self.board.shape

        for block_name in sorted(set(self.available_blocks)):
            block = TETRIMINOS[block_name]
            h, w = block.shape

            for x in range(H - h + 1):
                for y in range(W - w + 1):
                    if self.can_place(block_name, x, y):
                        actions.append((block_name, x, y))

        return actions

    def is_terminal(self):
        return self.is_goal() or len(self.get_valid_actions()) == 0

    def place_block(self, block_name, x, y, return_info=False):
        if block_name not in self.available_blocks:
            return (None, None) if return_info else None

        if not self.can_place(block_name, x, y):
            return (None, None) if return_info else None

        info = apply_block_and_clear(self.board, block_name, x, y)

        new_blocks = list(self.available_blocks)
        new_blocks.remove(block_name)

        new_state = State(
            board=info["board_after_clear"],
            available_blocks=new_blocks,
            current_score=self.current_score + info["cleared_cells"],
        )

        if return_info:
            return new_state, info
        return new_state

    def __repr__(self):
        return f"State(score={self.current_score}, remaining_blocks={self.available_blocks})"


class Node:
    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.priority = 0

        self.depth = 0
        if parent is not None:
            self.depth = parent.depth + 1

    def __lt__(self, other):
        return self.priority < other.priority
