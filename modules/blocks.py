"""Tetrimino shapes and their rotations.

Section 1 of the notebook. `TETRIMINOS` is the flattened dict of all rotations
(`I_0`, `I_90`, `S_0`..`S_270`, etc., 24 entries total). `O` has no rotations,
`I` has 2, the rest have 4. **Available-blocks lists must use TETRIMINOS keys,
not base names** — e.g. ``'I_0'`` not ``'I'``.
"""

import numpy as np


base = {
    'O': np.array([[1, 1],
                   [1, 1]]),
    'I': np.array([[1, 1, 1, 1]]),
    'S': np.array([[0, 1, 1],
                   [1, 1, 0]]),
    'Z': np.array([[1, 1, 0],
                   [0, 1, 1]]),
    'L': np.array([[0, 0, 1],
                   [1, 1, 1]]),
    'J': np.array([[1, 0, 0],
                   [1, 1, 1]]),
    'T': np.array([[0, 1, 0],
                   [1, 1, 1]]),
}

TETRIMINOS = {
    'O': base['O'],

    'I_0': base['I'],
    'I_90': np.rot90(base['I']),

    'S_0': base['S'],
    'S_90': np.rot90(base['S']),
    'S_180': np.rot90(base['S'], 2),
    'S_270': np.rot90(base['S'], 3),

    'Z_0': base['Z'],
    'Z_90': np.rot90(base['Z']),
    'Z_180': np.rot90(base['Z'], 2),
    'Z_270': np.rot90(base['Z'], 3),

    'L_0': base['L'],
    'L_90': np.rot90(base['L']),
    'L_180': np.rot90(base['L'], 2),
    'L_270': np.rot90(base['L'], 3),

    'J_0': base['J'],
    'J_90': np.rot90(base['J']),
    'J_180': np.rot90(base['J'], 2),
    'J_270': np.rot90(base['J'], 3),

    'T_0': base['T'],
    'T_90': np.rot90(base['T']),
    'T_180': np.rot90(base['T'], 2),
    'T_270': np.rot90(base['T'], 3),
}
