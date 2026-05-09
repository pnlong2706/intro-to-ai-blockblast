"""Board utilities: line clearing, connected-component flood fill, bounding box.

Pure functions over numpy boards (``int`` arrays where ``0`` is empty and ``1``
is filled). Section 2.1 of the notebook.
"""

import numpy as np

from .blocks import TETRIMINOS


def clear_full_lines(board):
    """
    Clear every fully occupied row and column using Numpy vectorization.

    Returns:
        new_board: board after clearing
        cleared_cells: number of cells removed (computed as the difference
            between board.sum() and new_board.sum())
        full_rows: list of full row indices
        full_cols: list of full column indices
    """
    full_rows = np.where(board.all(axis=1))[0].tolist()
    full_cols = np.where(board.all(axis=0))[0].tolist()

    new_board = np.copy(board)

    if full_rows:
        new_board[full_rows, :] = 0
    if full_cols:
        new_board[:, full_cols] = 0

    cleared_cells = int(board.sum() - new_board.sum())
    return new_board, cleared_cells, full_rows, full_cols


def get_empty_components(board):
    """
    Group connected empty cells (4-directional) into components.

    Returns a list of components, each component a list of (row, col) tuples.
    """
    H, W = board.shape
    visited = np.zeros((H, W), dtype=bool)
    components = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for x in range(H):
        for y in range(W):
            if board[x, y] != 0 or visited[x, y]:
                continue

            stack = [(x, y)]
            visited[x, y] = True
            cells = []

            while stack:
                cx, cy = stack.pop()
                cells.append((cx, cy))

                for dx, dy in directions:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < H and 0 <= ny < W:
                        if board[nx, ny] == 0 and not visited[nx, ny]:
                            visited[nx, ny] = True
                            stack.append((nx, ny))

            components.append(cells)

    return components


def count_empty_components(board):
    """Number of connected empty components."""
    return len(get_empty_components(board))


def bounding_box_area(board):
    """Area of the smallest rectangle covering all occupied cells."""
    occupied = np.argwhere(board == 1)

    if len(occupied) == 0:
        return 0

    min_x, min_y = occupied.min(axis=0)
    max_x, max_y = occupied.max(axis=0)

    return (max_x - min_x + 1) * (max_y - min_y + 1)


def apply_block_and_clear(board, block_name, x, y):
    """
    Place ``block_name`` at position ``(x, y)`` on ``board`` and clear any
    completed rows/columns afterwards.

    Returns a dict with the post-place board, post-clear board, placed cells,
    cleared cells count, list of full rows / cols, and total lines cleared.
    """
    block = TETRIMINOS[block_name]
    h, w = block.shape

    board_after_place = np.copy(board)
    placed_cells = []

    for i in range(h):
        for j in range(w):
            if block[i, j] == 1:
                board_after_place[x + i, y + j] = 1
                placed_cells.append((x + i, y + j))

    board_after_clear, cleared_cells, full_rows, full_cols = clear_full_lines(board_after_place)

    return {
        "board_after_place": board_after_place,
        "board_after_clear": board_after_clear,
        "placed_cells": placed_cells,
        "cleared_cells": cleared_cells,
        "full_rows": full_rows,
        "full_cols": full_cols,
        "lines_cleared": len(full_rows) + len(full_cols),
    }
