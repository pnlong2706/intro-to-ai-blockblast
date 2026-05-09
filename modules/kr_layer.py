"""Knowledge representation layer (Module C).

Implements the predicate logic from Chapter 4 of the report. The base
predicates (``Empty``, ``Remain``, ``Placeable``, ``Covers``, ``Line``) are
encoded directly in the existing State / board utilities; this module adds
the **derived** predicates that the search consults:

- ``Fillable(i,j,s)``  — implemented by :func:`compute_coverage_mask`
- ``Trapped(i,j,s)``   — :func:`has_trapped_cell`
- ``OneAway`` + ``Completes`` — :func:`action_completes_line`
"""

import numpy as np

from .blocks import TETRIMINOS


def compute_coverage_mask(state):
    """
    Build a board-shaped boolean mask where ``mask[i,j]`` is True iff there
    exists some ``(b, x, y)`` with ``Remain(b,s) AND Placeable(b,x,y,s) AND
    Covers(b,x,y,i,j)``.

    This realises ``Fillable`` as a single sweep per state:
        ``Fillable(i,j,s)  <=>  coverage_mask[i,j]``
    """
    H, W = state.board.shape
    mask = np.zeros((H, W), dtype=bool)

    for block_name in set(state.available_blocks):
        block = TETRIMINOS[block_name]
        h, w = block.shape

        for x in range(H - h + 1):
            for y in range(W - w + 1):
                if not state.can_place(block_name, x, y):
                    continue
                for i in range(h):
                    for j in range(w):
                        if block[i, j] == 1:
                            mask[x + i, y + j] = True

    return mask


def has_trapped_cell(state):
    """
    ``Trapped(i,j,s)  <=>  Empty(i,j,s) AND NOT Fillable(i,j,s)``

    Evaluated on ``state.board`` which, after a ``place_block`` call, is
    already ``board_after_clear`` (place_block runs ``apply_block_and_clear``
    internally). Returns ``False`` for the goal state since ``Trapped`` is
    only meaningful while there are still blocks to place this turn.
    """
    if len(state.available_blocks) == 0:
        return False

    coverage = compute_coverage_mask(state)
    empty = (state.board == 0)
    trapped = empty & ~coverage
    return bool(trapped.any())


def action_completes_line(state, action):
    """
    ``Completes(b,x,y,s)``  <=>  there exists a line ``ell`` and cell ``(i,j)``
    with ``OneAway(ell, s) AND (i,j) in ell AND Empty(i,j,s) AND
    Covers(b,x,y,i,j)``.

    A row/column is ``OneAway`` iff it has exactly one empty cell on the
    pre-placement board AND the action's covered cells include that empty
    cell.
    """
    block_name, x, y = action
    block = TETRIMINOS[block_name]
    H, W = state.board.shape
    h, w = block.shape

    covered = set()
    for i in range(h):
        for j in range(w):
            if block[i, j] == 1:
                covered.add((x + i, y + j))

    for r in range(H):
        empties = [j for j in range(W) if state.board[r, j] == 0]
        if len(empties) == 1 and (r, empties[0]) in covered:
            return True

    for c in range(W):
        empties = [i for i in range(H) if state.board[i, c] == 0]
        if len(empties) == 1 and (empties[0], c) in covered:
            return True

    return False
