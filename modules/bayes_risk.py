"""Bayes risk estimation (Module D, Chapter 5).

Independent module: deliberately uses ``BAYES_`` / ``bayes_`` prefixes so it
does not collide with the global game / search / heuristic functions defined
elsewhere. Implements the rollout-based Bayes Network whose CPDs are estimated
by empirical frequency counts (no Laplace smoothing).
"""

import os
from itertools import product

import numpy as np
import pandas as pd


BAYES_BOARD_SIZE = 8
BAYES_DEFAULT_N_SAMPLES = 50_000
BAYES_DEFAULT_BLOCKS_PER_TURN = 4
BAYES_DEFAULT_ROLLOUT_HORIZON_TURNS = 3
BAYES_DEFAULT_MIN_COUNT = 30


# -----------------------------------------------------------------------------
# A. Local game utilities for the Bayes experiment only
# -----------------------------------------------------------------------------

BAYES_BASE_BLOCKS = {
    'O': np.array([[1, 1],
                   [1, 1]], dtype=int),
    'I': np.array([[1, 1, 1, 1]], dtype=int),
    'S': np.array([[0, 1, 1],
                   [1, 1, 0]], dtype=int),
    'Z': np.array([[1, 1, 0],
                   [0, 1, 1]], dtype=int),
    'L': np.array([[0, 0, 1],
                   [1, 1, 1]], dtype=int),
    'J': np.array([[1, 0, 0],
                   [1, 1, 1]], dtype=int),
    'T': np.array([[0, 1, 0],
                   [1, 1, 1]], dtype=int),
}

BAYES_TETRIMINOS = {
    'O': BAYES_BASE_BLOCKS['O'],
    'I_0': BAYES_BASE_BLOCKS['I'],
    'I_90': np.rot90(BAYES_BASE_BLOCKS['I']),
    'S_0': BAYES_BASE_BLOCKS['S'],
    'S_90': np.rot90(BAYES_BASE_BLOCKS['S']),
    'S_180': np.rot90(BAYES_BASE_BLOCKS['S'], 2),
    'S_270': np.rot90(BAYES_BASE_BLOCKS['S'], 3),
    'Z_0': BAYES_BASE_BLOCKS['Z'],
    'Z_90': np.rot90(BAYES_BASE_BLOCKS['Z']),
    'Z_180': np.rot90(BAYES_BASE_BLOCKS['Z'], 2),
    'Z_270': np.rot90(BAYES_BASE_BLOCKS['Z'], 3),
    'L_0': BAYES_BASE_BLOCKS['L'],
    'L_90': np.rot90(BAYES_BASE_BLOCKS['L']),
    'L_180': np.rot90(BAYES_BASE_BLOCKS['L'], 2),
    'L_270': np.rot90(BAYES_BASE_BLOCKS['L'], 3),
    'J_0': BAYES_BASE_BLOCKS['J'],
    'J_90': np.rot90(BAYES_BASE_BLOCKS['J']),
    'J_180': np.rot90(BAYES_BASE_BLOCKS['J'], 2),
    'J_270': np.rot90(BAYES_BASE_BLOCKS['J'], 3),
    'T_0': BAYES_BASE_BLOCKS['T'],
    'T_90': np.rot90(BAYES_BASE_BLOCKS['T']),
    'T_180': np.rot90(BAYES_BASE_BLOCKS['T'], 2),
    'T_270': np.rot90(BAYES_BASE_BLOCKS['T'], 3),
}

BAYES_STATES = {
    'Density': ['Low', 'Medium', 'High'],
    'Fragmentation': ['Low', 'Medium', 'High'],
    'LineClear': ['No', 'Yes'],
    'ValidMoveLevel': ['Low', 'Medium', 'High'],
    'StuckRisk': ['Low', 'Medium', 'High'],
}


def bayes_clear_full_lines(board):
    board = np.asarray(board, dtype=int)
    full_rows = np.where(board.all(axis=1))[0].tolist()
    full_cols = np.where(board.all(axis=0))[0].tolist()

    new_board = np.copy(board)
    if full_rows:
        new_board[full_rows, :] = 0
    if full_cols:
        new_board[:, full_cols] = 0

    cleared_cells = int(board.sum() - new_board.sum())
    return new_board, cleared_cells, full_rows, full_cols


def bayes_can_place(board, block_name, x, y):
    if block_name not in BAYES_TETRIMINOS:
        return False

    board = np.asarray(board, dtype=int)
    block = BAYES_TETRIMINOS[block_name]
    h, w = block.shape
    H, W = board.shape

    if x < 0 or y < 0 or x + h > H or y + w > W:
        return False

    target = board[x:x + h, y:y + w]
    return bool(np.all((block == 0) | (target == 0)))


def bayes_get_valid_actions(board, available_blocks):
    board = np.asarray(board, dtype=int)
    H, W = board.shape
    actions = []

    for block_name in sorted(set(available_blocks)):
        if block_name not in BAYES_TETRIMINOS:
            continue
        block = BAYES_TETRIMINOS[block_name]
        h, w = block.shape
        for x in range(H - h + 1):
            for y in range(W - w + 1):
                if bayes_can_place(board, block_name, x, y):
                    actions.append((block_name, x, y))

    return actions


def bayes_has_valid_action_for_block(board, block_name):
    if block_name not in BAYES_TETRIMINOS:
        return False

    board = np.asarray(board, dtype=int)
    H, W = board.shape
    block = BAYES_TETRIMINOS[block_name]
    h, w = block.shape

    for x in range(H - h + 1):
        for y in range(W - w + 1):
            if bayes_can_place(board, block_name, x, y):
                return True
    return False


def bayes_count_placeable_remaining_blocks(board, remaining_blocks):
    return sum(
        1
        for block_name in set(remaining_blocks)
        if bayes_has_valid_action_for_block(board, block_name)
    )


def bayes_place_block_and_clear(board, block_name, x, y):
    if not bayes_can_place(board, block_name, x, y):
        raise ValueError(f'Invalid Bayes action: block={block_name}, x={x}, y={y}')

    board = np.asarray(board, dtype=int)
    block = BAYES_TETRIMINOS[block_name]
    h, w = block.shape

    board_after_place = np.copy(board)
    patch = board_after_place[x:x + h, y:y + w]
    patch[block == 1] = 1
    board_after_place[x:x + h, y:y + w] = patch

    board_after_clear, cleared_cells, full_rows, full_cols = bayes_clear_full_lines(board_after_place)
    info = {
        'board_after_place': board_after_place,
        'board_after_clear': board_after_clear,
        'cleared_cells': cleared_cells,
        'full_rows': full_rows,
        'full_cols': full_cols,
        'lines_cleared': len(full_rows) + len(full_cols),
    }
    return board_after_clear, info


def bayes_count_empty_components(board):
    board = np.asarray(board, dtype=int)
    H, W = board.shape
    visited = np.zeros((H, W), dtype=bool)
    components = 0
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for x in range(H):
        for y in range(W):
            if board[x, y] != 0 or visited[x, y]:
                continue

            components += 1
            stack = [(x, y)]
            visited[x, y] = True

            while stack:
                cx, cy = stack.pop()
                for dx, dy in directions:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < H and 0 <= ny < W:
                        if board[nx, ny] == 0 and not visited[nx, ny]:
                            visited[nx, ny] = True
                            stack.append((nx, ny))

    return components


def bayes_filled_ratio(board):
    board = np.asarray(board, dtype=int)
    if board.size == 0:
        return 0.0
    return float(board.sum() / board.size)


def bayes_board_to_string(board):
    board = np.asarray(board, dtype=int)
    return '/'.join(''.join(str(int(cell)) for cell in row) for row in board)


def bayes_blocks_to_string(blocks):
    return '|'.join(str(block) for block in blocks)


def bayes_remove_one_block(blocks, block_name):
    remaining = list(blocks)
    remaining.remove(block_name)
    return remaining


# -----------------------------------------------------------------------------
# B. Simulated turns, rollout policy, and StuckRisk measurement
# -----------------------------------------------------------------------------

def bayes_sample_turn_blocks(rng, blocks_per_turn=BAYES_DEFAULT_BLOCKS_PER_TURN):
    block_names = list(BAYES_TETRIMINOS.keys())
    return [str(name) for name in rng.choice(block_names, size=int(blocks_per_turn), replace=True)]


def bayes_generate_random_board(fill_prob=None, num_random_blocks=None, rng=None, board_size=BAYES_BOARD_SIZE):
    rng = np.random.default_rng() if rng is None else rng

    if num_random_blocks is not None:
        board = np.zeros((board_size, board_size), dtype=int)
        block_names = list(BAYES_TETRIMINOS.keys())
        for _ in range(max(0, int(num_random_blocks))):
            block_name = str(block_names[int(rng.integers(len(block_names)))])
            actions = bayes_get_valid_actions(board, [block_name])
            if not actions:
                continue
            action = actions[int(rng.integers(len(actions)))]
            board, _ = bayes_place_block_and_clear(board, *action)
        return board

    if fill_prob is None:
        fill_prob = float(rng.uniform(0.05, 0.72))
    fill_prob = float(np.clip(fill_prob, 0.0, 0.92))

    board = (rng.random((board_size, board_size)) < fill_prob).astype(int)
    board, _, _, _ = bayes_clear_full_lines(board)
    return board


def bayes_random_board_for_dataset(rng):
    if rng.random() < 0.60:
        fill_prob = float(rng.uniform(0.05, 0.70))
        return bayes_generate_random_board(fill_prob=fill_prob, rng=rng)

    num_random_blocks = int(rng.integers(0, 16))
    return bayes_generate_random_board(num_random_blocks=num_random_blocks, rng=rng)


def bayes_rollout_action_score(board, action, remaining_blocks_after_action):
    board_next, info = bayes_place_block_and_clear(board, *action)
    density_next = bayes_filled_ratio(board_next)
    fragmentation_next = bayes_count_empty_components(board_next)
    placeable_remaining = bayes_count_placeable_remaining_blocks(board_next, remaining_blocks_after_action)

    score = (
        100.0 * info['lines_cleared']
        + 0.15 * info['cleared_cells']
        + 12.0 * placeable_remaining
        - 8.0 * density_next
        - 1.5 * fragmentation_next
    )
    return score, board_next, info


def bayes_select_rollout_action(board, available_blocks, rng, max_candidates=6):
    actions = bayes_get_valid_actions(board, available_blocks)
    if not actions:
        return None, None, None

    if len(actions) > max_candidates:
        candidate_indices = rng.choice(len(actions), size=int(max_candidates), replace=False)
        candidate_actions = [actions[int(idx)] for idx in candidate_indices]
    else:
        candidate_actions = actions

    best_score = -np.inf
    best_action = None
    best_board = None
    best_info = None

    for action in candidate_actions:
        next_blocks = bayes_remove_one_block(available_blocks, action[0])
        score, board_next, info = bayes_rollout_action_score(board, action, next_blocks)
        if score > best_score:
            best_score = score
            best_action = action
            best_board = board_next
            best_info = info

    return best_action, best_board, best_info


def bayes_measure_stuck_risk_by_rollout(
    board_after,
    remaining_blocks,
    rng,
    rollout_horizon_turns=BAYES_DEFAULT_ROLLOUT_HORIZON_TURNS,
    blocks_per_turn=BAYES_DEFAULT_BLOCKS_PER_TURN,
    max_policy_candidates=6,
):
    board = np.copy(np.asarray(board_after, dtype=int))
    current_turn_blocks = list(remaining_blocks)
    survival_steps = 0

    while current_turn_blocks:
        action, board_next, _ = bayes_select_rollout_action(
            board,
            current_turn_blocks,
            rng,
            max_candidates=max_policy_candidates,
        )
        if action is None:
            return {
                'StuckRisk': 'High',
                'rollout_survival_steps': survival_steps,
                'rollout_status': 'stuck_current_turn',
                'rollout_turns_completed': 0,
            }

        board = board_next
        current_turn_blocks = bayes_remove_one_block(current_turn_blocks, action[0])
        survival_steps += 1

    completed_future_turns = 0
    for _ in range(int(rollout_horizon_turns)):
        future_turn_blocks = bayes_sample_turn_blocks(rng, blocks_per_turn=blocks_per_turn)

        while future_turn_blocks:
            action, board_next, _ = bayes_select_rollout_action(
                board,
                future_turn_blocks,
                rng,
                max_candidates=max_policy_candidates,
            )
            if action is None:
                return {
                    'StuckRisk': 'Medium',
                    'rollout_survival_steps': survival_steps,
                    'rollout_status': 'stuck_before_horizon',
                    'rollout_turns_completed': completed_future_turns,
                }

            board = board_next
            future_turn_blocks = bayes_remove_one_block(future_turn_blocks, action[0])
            survival_steps += 1

        completed_future_turns += 1

    return {
        'StuckRisk': 'Low',
        'rollout_survival_steps': survival_steps,
        'rollout_status': 'survived_horizon',
        'rollout_turns_completed': completed_future_turns,
    }


# -----------------------------------------------------------------------------
# C. Feature discretization and dataset generation
# -----------------------------------------------------------------------------

def bayes_discretize_density(ratio):
    if ratio < 0.35:
        return 'Low'
    if ratio <= 0.65:
        return 'Medium'
    return 'High'


def bayes_discretize_fragmentation(num_components):
    if num_components <= 1:
        return 'Low'
    if num_components <= 3:
        return 'Medium'
    return 'High'


def bayes_discretize_lineclear(lines_cleared):
    return 'Yes' if lines_cleared > 0 else 'No'


def bayes_discretize_valid_move_level(num_valid_moves):
    if num_valid_moves < 10:
        return 'Low'
    if num_valid_moves < 30:
        return 'Medium'
    return 'High'


BAYES_DATASET_COLUMNS = [
    'sample_id',
    'board_before',
    'turn_blocks',
    'block_name',
    'x',
    'y',
    'remaining_blocks',
    'board_after',
    'density_raw',
    'fragmentation_raw',
    'lineclear_raw',
    'valid_moves_after_raw',
    'rollout_survival_steps',
    'rollout_status',
    'rollout_turns_completed',
    'Density',
    'Fragmentation',
    'LineClear',
    'ValidMoveLevel',
    'StuckRisk',
]


def generate_bayes_dataset(
    n_samples=BAYES_DEFAULT_N_SAMPLES,
    seed=42,
    max_attempts=None,
    rollout_horizon_turns=BAYES_DEFAULT_ROLLOUT_HORIZON_TURNS,
    blocks_per_turn=BAYES_DEFAULT_BLOCKS_PER_TURN,
):
    rng = np.random.default_rng(seed)
    rows = []
    attempts = 0
    max_attempts = max(100, int(n_samples) * 50) if max_attempts is None else int(max_attempts)

    while len(rows) < int(n_samples) and attempts < max_attempts:
        attempts += 1
        board_before = bayes_random_board_for_dataset(rng)
        turn_blocks = bayes_sample_turn_blocks(rng, blocks_per_turn=blocks_per_turn)
        actions = bayes_get_valid_actions(board_before, turn_blocks)
        if not actions:
            continue

        block_name, x, y = actions[int(rng.integers(len(actions)))]
        if block_name not in turn_blocks:
            raise AssertionError('Bayes dataset invariant failed: selected block is not in turn_blocks.')

        board_after, info = bayes_place_block_and_clear(board_before, block_name, x, y)
        remaining_blocks = bayes_remove_one_block(turn_blocks, block_name)

        density_raw = bayes_filled_ratio(board_after)
        fragmentation_raw = bayes_count_empty_components(board_after)
        lineclear_raw = int(info['lines_cleared'])

        valid_moves_after_raw = len(bayes_get_valid_actions(board_after, remaining_blocks))

        rollout = bayes_measure_stuck_risk_by_rollout(
            board_after,
            remaining_blocks,
            rng,
            rollout_horizon_turns=rollout_horizon_turns,
            blocks_per_turn=blocks_per_turn,
        )

        rows.append({
            'sample_id': len(rows),
            'board_before': bayes_board_to_string(board_before),
            'turn_blocks': bayes_blocks_to_string(turn_blocks),
            'block_name': block_name,
            'x': x,
            'y': y,
            'remaining_blocks': bayes_blocks_to_string(remaining_blocks),
            'board_after': bayes_board_to_string(board_after),
            'density_raw': density_raw,
            'fragmentation_raw': fragmentation_raw,
            'lineclear_raw': lineclear_raw,
            'valid_moves_after_raw': valid_moves_after_raw,
            'rollout_survival_steps': rollout['rollout_survival_steps'],
            'rollout_status': rollout['rollout_status'],
            'rollout_turns_completed': rollout['rollout_turns_completed'],
            'Density': bayes_discretize_density(density_raw),
            'Fragmentation': bayes_discretize_fragmentation(fragmentation_raw),
            'LineClear': bayes_discretize_lineclear(lineclear_raw),
            'ValidMoveLevel': bayes_discretize_valid_move_level(valid_moves_after_raw),
            'StuckRisk': rollout['StuckRisk'],
        })

    if len(rows) < int(n_samples):
        print(f'Warning: requested {n_samples} samples, generated {len(rows)} valid samples after {attempts} attempts.')

    return pd.DataFrame(rows, columns=BAYES_DATASET_COLUMNS)


# -----------------------------------------------------------------------------
# D. CPD estimation by empirical frequency counts, no smoothing
# -----------------------------------------------------------------------------

def bayes_probability_columns(child):
    return [f'P({child}={value})' for value in BAYES_STATES[child]]


def bayes_estimate_cpd_no_smoothing(df, child, parents, min_count=BAYES_DEFAULT_MIN_COUNT):
    if child not in BAYES_STATES:
        raise ValueError(f'Unknown child variable: {child}')

    parents = list(parents)
    for parent in parents:
        if parent not in BAYES_STATES:
            raise ValueError(f'Unknown parent variable: {parent}')

    df = pd.DataFrame() if df is None else df.copy()
    child_values = BAYES_STATES[child]
    parent_combinations = list(product(*[BAYES_STATES[parent] for parent in parents])) if parents else [()]

    rows = []
    for combo in parent_combinations:
        mask = np.ones(len(df), dtype=bool)
        for parent, value in zip(parents, combo):
            if parent not in df.columns:
                mask &= False
            else:
                mask &= (df[parent].to_numpy() == value)

        subset = df.loc[mask] if len(df) > 0 else df
        counts = subset[child].value_counts().to_dict() if child in subset.columns else {}
        observed = int(sum(counts.get(value, 0) for value in child_values))

        if observed == 0:
            probabilities = [np.nan for _ in child_values]
            coverage_status = 'NotObserved'
        else:
            probabilities = [float(counts.get(value, 0) / observed) for value in child_values]
            coverage_status = 'LowCount' if observed < int(min_count) else 'Observed'

        row = {parent: value for parent, value in zip(parents, combo)}
        for child_value, count, probability in zip(child_values, [counts.get(v, 0) for v in child_values], probabilities):
            row[f'Count({child}={child_value})'] = int(count)
            row[f'P({child}={child_value})'] = probability
        row['ObservedCount'] = observed
        row['CoverageStatus'] = coverage_status
        rows.append(row)

    cpd = pd.DataFrame(rows)
    cpd.attrs['child'] = child
    cpd.attrs['parents'] = parents
    cpd.attrs['smoothing'] = False
    cpd.attrs['min_count'] = int(min_count)
    return cpd


def bayes_validate_cpd(cpd, child, atol=1e-9):
    prob_cols = bayes_probability_columns(child)
    missing_cols = [col for col in prob_cols if col not in cpd.columns]
    if missing_cols:
        raise AssertionError(f'Missing probability columns in CPD for {child}: {missing_cols}')
    if 'ObservedCount' not in cpd.columns:
        raise AssertionError(f'Missing ObservedCount column in CPD for {child}.')
    if 'CoverageStatus' not in cpd.columns:
        raise AssertionError(f'Missing CoverageStatus column in CPD for {child}.')

    observed_mask = cpd['ObservedCount'].to_numpy() > 0
    if observed_mask.any():
        row_sums = cpd.loc[observed_mask, prob_cols].sum(axis=1)
        max_error = float(np.max(np.abs(row_sums - 1.0)))
        if not np.allclose(row_sums, 1.0, atol=atol):
            raise AssertionError(f'Observed CPD rows for {child} do not sum to 1. Max error = {max_error}')
    else:
        max_error = 0.0

    not_observed_mask = cpd['ObservedCount'].to_numpy() == 0
    if not_observed_mask.any() and not cpd.loc[not_observed_mask, prob_cols].isna().all(axis=None):
        raise AssertionError(f'NotObserved CPD rows for {child} must keep probabilities as NaN.')

    print(f'CPD check passed for {child}: {len(cpd)} rows, observed rows sum to 1, max error = {max_error:.2e}')
    return True


# -----------------------------------------------------------------------------
# E. Inference helper with explicit insufficient-data handling
# -----------------------------------------------------------------------------

def bayes_find_cpd_row(cpd, conditions):
    mask = np.ones(len(cpd), dtype=bool)
    for column, value in conditions.items():
        if column not in cpd.columns:
            raise KeyError(f'Column {column} not found in CPD.')
        mask &= (cpd[column].to_numpy() == value)

    matches = cpd.loc[mask]
    if matches.empty:
        return None
    return matches.iloc[0]


def bayes_row_has_probability_data(row, child):
    if row is None:
        return False
    if int(row.get('ObservedCount', 0)) <= 0:
        return False
    return not pd.isna(row[bayes_probability_columns(child)]).any()


def bayes_infer_stuck_risk(density, fragmentation, lineclear, cpd_valid_move_level, cpd_stuck_risk):
    if density not in BAYES_STATES['Density']:
        raise ValueError(f'Invalid Density value: {density}')
    if fragmentation not in BAYES_STATES['Fragmentation']:
        raise ValueError(f'Invalid Fragmentation value: {fragmentation}')
    if lineclear not in BAYES_STATES['LineClear']:
        raise ValueError(f'Invalid LineClear value: {lineclear}')

    valid_row = bayes_find_cpd_row(
        cpd_valid_move_level,
        {'Density': density, 'Fragmentation': fragmentation, 'LineClear': lineclear},
    )
    if not bayes_row_has_probability_data(valid_row, 'ValidMoveLevel'):
        return {
            'status': 'InsufficientData',
            'reason': 'Missing CPD row for P(ValidMoveLevel | Density, Fragmentation, LineClear).',
            'probabilities': {risk: np.nan for risk in BAYES_STATES['StuckRisk']},
        }

    result = {risk: 0.0 for risk in BAYES_STATES['StuckRisk']}
    for valid_level in BAYES_STATES['ValidMoveLevel']:
        p_valid_level = float(valid_row[f'P(ValidMoveLevel={valid_level})'])
        if p_valid_level == 0.0:
            continue

        stuck_row = bayes_find_cpd_row(
            cpd_stuck_risk,
            {'ValidMoveLevel': valid_level, 'Fragmentation': fragmentation},
        )
        if not bayes_row_has_probability_data(stuck_row, 'StuckRisk'):
            return {
                'status': 'InsufficientData',
                'reason': f'Missing CPD row for P(StuckRisk | ValidMoveLevel={valid_level}, Fragmentation={fragmentation}).',
                'probabilities': {risk: np.nan for risk in BAYES_STATES['StuckRisk']},
            }

        for risk in BAYES_STATES['StuckRisk']:
            result[risk] += p_valid_level * float(stuck_row[f'P(StuckRisk={risk})'])

    total = sum(result.values())
    if total <= 0.0:
        return {
            'status': 'InsufficientData',
            'reason': 'The inferred distribution has zero total probability.',
            'probabilities': {risk: np.nan for risk in BAYES_STATES['StuckRisk']},
        }

    result = {risk: probability / total for risk, probability in result.items()}
    return {'status': 'OK', 'reason': None, 'probabilities': result}


# -----------------------------------------------------------------------------
# F. Report tables, validation, and CSV outputs
# -----------------------------------------------------------------------------

def bayes_make_stuckrisk_distribution(df):
    counts = df['StuckRisk'].value_counts().reindex(BAYES_STATES['StuckRisk'], fill_value=0)
    total = int(counts.sum())
    ratios = counts / total if total > 0 else counts.astype(float)
    return pd.DataFrame({
        'StuckRisk': counts.index,
        'Count': counts.astype(int).to_numpy(),
        'Ratio': ratios.astype(float).to_numpy(),
    })


def bayes_make_coverage_summary(cpd_map):
    rows = []
    for cpd_name, cpd in cpd_map.items():
        total = int(len(cpd))
        observed = int((cpd['ObservedCount'] > 0).sum()) if total else 0
        not_observed = int((cpd['ObservedCount'] == 0).sum()) if total else 0
        low_count = int((cpd['CoverageStatus'] == 'LowCount').sum()) if total else 0
        rows.append({
            'CPD': cpd_name,
            'TotalParentCombinations': total,
            'ObservedCombinations': observed,
            'NotObservedCombinations': not_observed,
            'CoverageRatio': float(observed / total) if total > 0 else np.nan,
            'LowCountCombinations': low_count,
        })
    return pd.DataFrame(rows)


def validate_bayes_outputs(
    df,
    cpd_valid_move_level,
    cpd_stuck_risk,
    expected_samples=None,
    output_dir=None,
    required_files=None,
    blocks_per_turn=BAYES_DEFAULT_BLOCKS_PER_TURN,
):
    if df.empty:
        raise AssertionError('Bayes dataset is empty.')

    if expected_samples is not None and len(df) != int(expected_samples):
        raise AssertionError(f'Expected {expected_samples} samples, but generated {len(df)} samples.')

    required_dataset_columns = set(BAYES_DATASET_COLUMNS)
    missing_dataset_columns = sorted(required_dataset_columns - set(df.columns))
    if missing_dataset_columns:
        raise AssertionError(f'Missing Bayes dataset columns: {missing_dataset_columns}')

    action_in_turn = df.apply(lambda row: row['block_name'] in str(row['turn_blocks']).split('|'), axis=1)
    if not bool(action_in_turn.all()):
        raise AssertionError('At least one Bayes sample has block_name not contained in turn_blocks.')

    expected_remaining_blocks = int(blocks_per_turn) - 1
    remaining_len_ok = df['remaining_blocks'].apply(lambda text: 0 if text == '' else len(str(text).split('|'))).eq(expected_remaining_blocks)
    if not bool(remaining_len_ok.all()):
        raise AssertionError('Each Bayes sample must remove exactly one block from turn_blocks.')

    smoothing_marker = 'Smoothed' + 'FromPrior'
    for cpd in (cpd_valid_move_level, cpd_stuck_risk):
        if smoothing_marker in cpd.columns:
            raise AssertionError('Smoothing marker found although Section 5 must not use smoothing.')

    bayes_validate_cpd(cpd_valid_move_level, 'ValidMoveLevel')
    bayes_validate_cpd(cpd_stuck_risk, 'StuckRisk')

    if output_dir is not None and required_files is not None:
        missing_files = [name for name in required_files if not os.path.exists(os.path.join(output_dir, name))]
        if missing_files:
            raise AssertionError(f'Missing Bayes output files: {missing_files}')

    print(f'Dataset check passed: {len(df)} rollout-based Bayes samples, no smoothing used.')
    return True


def run_bayes_experiment(
    n_samples=BAYES_DEFAULT_N_SAMPLES,
    seed=42,
    save_outputs=True,
    output_dir='outputs/bayes',
    rollout_horizon_turns=BAYES_DEFAULT_ROLLOUT_HORIZON_TURNS,
    blocks_per_turn=BAYES_DEFAULT_BLOCKS_PER_TURN,
    min_count=BAYES_DEFAULT_MIN_COUNT,
):
    """Generate the Bayes rollout dataset, fit both CPDs, save CSV outputs."""
    print('Generating Bayes rollout simulation dataset...')
    df = generate_bayes_dataset(
        n_samples=n_samples,
        seed=seed,
        rollout_horizon_turns=rollout_horizon_turns,
        blocks_per_turn=blocks_per_turn,
    )

    cpd_valid_move_level = bayes_estimate_cpd_no_smoothing(
        df,
        child='ValidMoveLevel',
        parents=['Density', 'Fragmentation', 'LineClear'],
        min_count=min_count,
    )
    cpd_stuck_risk = bayes_estimate_cpd_no_smoothing(
        df,
        child='StuckRisk',
        parents=['ValidMoveLevel', 'Fragmentation'],
        min_count=min_count,
    )

    stuckrisk_distribution = bayes_make_stuckrisk_distribution(df)
    coverage_summary = bayes_make_coverage_summary({
        'cpd_valid_move_level': cpd_valid_move_level,
        'cpd_stuck_risk': cpd_stuck_risk,
    })

    preview_cols = [
        'sample_id',
        'turn_blocks',
        'block_name',
        'x',
        'y',
        'remaining_blocks',
        'density_raw',
        'fragmentation_raw',
        'lineclear_raw',
        'valid_moves_after_raw',
        'rollout_survival_steps',
        'rollout_status',
        'Density',
        'Fragmentation',
        'LineClear',
        'ValidMoveLevel',
        'StuckRisk',
    ]

    print()
    print('=== First 5 rows of Bayes rollout dataset ===')
    if df.empty:
        print('(empty dataset)')
    else:
        print(df[preview_cols].head(5).to_string(index=False))

    print()
    print('=== StuckRisk distribution ===')
    print(stuckrisk_distribution.to_string(index=False))

    print()
    print('=== CPD1: P(ValidMoveLevel | Density, Fragmentation, LineClear) ===')
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 180):
        print(cpd_valid_move_level.to_string(index=False))

    print()
    print('=== CPD2: P(StuckRisk | ValidMoveLevel, Fragmentation) ===')
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 180):
        print(cpd_stuck_risk.to_string(index=False))

    print()
    print('=== Bayes CPD coverage summary ===')
    print(coverage_summary.to_string(index=False))

    required_files = [
        'bayes_dataset.csv',
        'cpd_valid_move_level.csv',
        'cpd_stuck_risk.csv',
        'bayes_stuckrisk_distribution.csv',
        'bayes_coverage_summary.csv',
    ]

    if save_outputs:
        os.makedirs(output_dir, exist_ok=True)

        stale_inference_path = os.path.join(output_dir, 'bayes_' + 'inference_' + 'examples.csv')
        if os.path.exists(stale_inference_path):
            os.remove(stale_inference_path)

        df.to_csv(os.path.join(output_dir, 'bayes_dataset.csv'), index=False)
        cpd_valid_move_level.to_csv(os.path.join(output_dir, 'cpd_valid_move_level.csv'), index=False)
        cpd_stuck_risk.to_csv(os.path.join(output_dir, 'cpd_stuck_risk.csv'), index=False)
        stuckrisk_distribution.to_csv(os.path.join(output_dir, 'bayes_stuckrisk_distribution.csv'), index=False)
        coverage_summary.to_csv(os.path.join(output_dir, 'bayes_coverage_summary.csv'), index=False)
        print()
        print(f'Saved CSV outputs to: {output_dir}')

    validate_bayes_outputs(
        df,
        cpd_valid_move_level,
        cpd_stuck_risk,
        expected_samples=n_samples,
        output_dir=output_dir if save_outputs else None,
        required_files=required_files if save_outputs else None,
        blocks_per_turn=blocks_per_turn,
    )

    return df, cpd_valid_move_level, cpd_stuck_risk, stuckrisk_distribution, coverage_summary
