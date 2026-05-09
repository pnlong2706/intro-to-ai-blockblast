"""Command-line entry point for the BlockBlast AI project.

Usage::

    python main.py demo          # one-turn KR ablation (~30 s)
    python main.py multi-turn    # 3-turn fixed game with baseline search
    python main.py bayes         # small Bayes rollout experiment (~1 min)
    python main.py --help

The notebook in ``notebooks/BlockBlast.ipynb`` is still the canonical
artefact for the assignment — this script is a convenience runner that
exercises the modules in ``modules/`` from the command line.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from modules import (
    State,
    best_first_search,
    print_solution,
    reconstruct_node_chain,
)


# ---------------------------------------------------------------------------
# Demo: one-turn KR ablation (Module A + B + C)
# ---------------------------------------------------------------------------

def cmd_demo(args: argparse.Namespace) -> int:
    initial_state = State(
        board=np.zeros((8, 8), dtype=int),
        available_blocks=["O", "I_0", "T_180", "L_90"],
        current_score=0,
    )

    print("=" * 60)
    print("DEMO — One-turn search, KR layer ablation")
    print("=" * 60)
    print(f"Empty 8x8 board, blocks: {initial_state.available_blocks}")
    print(f"max_expansions = {args.max_expansions}\n")

    print("--- KR layer OFF (pure baseline) ---")
    t0 = time.time()
    result_off = best_first_search(
        initial_state,
        max_expansions=args.max_expansions,
        prune_trapped=False,
        completion_bonus=0.0,
    )
    print_solution(result_off)
    print(f"Time: {time.time() - t0:.2f}s\n")

    print("--- KR layer ON (default) ---")
    t0 = time.time()
    result_on = best_first_search(
        initial_state,
        max_expansions=args.max_expansions,
    )
    print_solution(result_on)
    print(f"Time: {time.time() - t0:.2f}s")

    return 0


# ---------------------------------------------------------------------------
# Multi-turn game with baseline search
# ---------------------------------------------------------------------------

DEFAULT_TURNS_BLOCKS = [
    ["O", "I_0", "T_180", "L_90"],
    ["S_0", "Z_90", "O", "J_180"],
    ["T_0", "I_90", "L_0", "O"],
]


def cmd_multi_turn(args: argparse.Namespace) -> int:
    current_board = np.zeros((8, 8), dtype=int)
    total_score = 0

    print("=" * 60)
    print("MULTI-TURN — 3 fixed turns with baseline best_first_search")
    print("=" * 60)

    for turn_idx, blocks in enumerate(DEFAULT_TURNS_BLOCKS, start=1):
        s = State(board=current_board, available_blocks=blocks, current_score=0)
        t0 = time.time()
        node, expansions, terminal = best_first_search(
            s, max_expansions=args.max_expansions,
        )
        elapsed = time.time() - t0

        chain = reconstruct_node_chain(node)
        print(
            f"\nTurn {turn_idx}: blocks={blocks} -> "
            f"score={node.state.current_score} terminal={terminal} "
            f"expansions={expansions} time={elapsed:.2f}s"
        )
        for step, child in enumerate(chain[1:], start=1):
            block, x, y = child.action
            print(f"  step {step}: place {block} at ({x}, {y})")

        current_board = np.copy(node.state.board)
        total_score += node.state.current_score
        if not terminal:
            print(f"Turn {turn_idx} stopped before completion. Aborting.")
            break

    print(f"\nFinal total score across {turn_idx} turn(s): {total_score}")
    return 0


# ---------------------------------------------------------------------------
# Bayes rollout experiment (Module D)
# ---------------------------------------------------------------------------

def cmd_bayes(args: argparse.Namespace) -> int:
    from modules.bayes_risk import run_bayes_experiment

    print("=" * 60)
    print(f"BAYES — rollout experiment with n_samples={args.n_samples}")
    print("=" * 60)
    print(
        "Note: the report uses n_samples=50_000. This CLI defaults to a smaller "
        "value so the run finishes quickly; pass --n-samples to change it."
    )
    print()

    run_bayes_experiment(
        n_samples=args.n_samples,
        seed=args.seed,
        save_outputs=args.save_outputs,
        output_dir=args.output_dir,
    )
    return 0


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BlockBlast AI — CLI runner for the modules package.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_demo = sub.add_parser("demo", help="One-turn KR ablation demo (~30 s).")
    p_demo.add_argument("--max-expansions", type=int, default=1000)
    p_demo.set_defaults(func=cmd_demo)

    p_mt = sub.add_parser(
        "multi-turn", help="3 fixed turns with baseline best_first_search.",
    )
    p_mt.add_argument("--max-expansions", type=int, default=1500)
    p_mt.set_defaults(func=cmd_multi_turn)

    p_bayes = sub.add_parser("bayes", help="Bayes rollout experiment (Module D).")
    p_bayes.add_argument("--n-samples", type=int, default=2000)
    p_bayes.add_argument("--seed", type=int, default=42)
    p_bayes.add_argument("--no-save", dest="save_outputs", action="store_false")
    p_bayes.add_argument("--output-dir", type=str, default="outputs/bayes")
    p_bayes.set_defaults(func=cmd_bayes, save_outputs=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
