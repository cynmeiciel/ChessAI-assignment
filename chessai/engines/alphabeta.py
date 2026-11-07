"""Minimax and alpha-beta search. WRITTEN BY THE TEAM.

Work through the tutorials in order and make each step test pass:

    docs/learn/01-minimax.md           negamax                      tests/learn/test_step_01.py
    docs/learn/03-alpha-beta.md        alphabeta, root_search       tests/learn/test_step_03.py
    docs/learn/04-strong-alpha-beta.md order_moves, quiescence,     tests/learn/test_step_04.py
                                       the two engines

Conventions (the tests rely on them):
- Scores are centipawns from the SIDE TO MOVE's point of view (negamax form).
- Before looking at the depth, check `board.is_game_over()`: a checkmated
  side to move scores -(MATE_SCORE + depth); any other game end scores 0.
  (More depth left means the mate came sooner, so sooner mates score higher.)
- At depth 0, return `evaluate(board)`.
- Every function must leave `board` exactly as it found it (push, then pop).
"""
from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

import chess

from chessai.engines.base import SearchInfo, SearchLimit, SearchResult
from chessai.engines.evaluate import EVAL_NAME, EVAL_VERSION, MATE_SCORE, PIECE_VALUES, evaluate

INFINITY = 1_000_000            # bigger than any score, including mates
# AlphaBetaEngine passes this as `quiesce`. Set it to False when you benchmark
# minimax against alpha-beta, so that both search exactly the same tree.
USE_QUIESCENCE = True
DEFAULT_DEPTH = 3               # when a search has neither a depth nor a time limit
MAX_DEPTH = 64                  # iterative deepening never goes past this
Evaluate = Callable[[chess.Board], int]


class SearchTimeout(Exception):
    """Raise this from deep inside a search when the deadline passes or `stop` is set."""


@dataclass
class SearchControl:
    """What a running search shares with every node: when to give up, and a node count.

    root_search creates one; plain calls (what the step tests make) pass none
    and then nothing is timed or counted.
    """

    deadline: float | None = None          # a time.perf_counter() value
    stop: threading.Event | None = None
    nodes: int = 0                         # calls to negamax / alphabeta / quiescence

    def visit(self) -> None:
        """Count one node; raise SearchTimeout if the time is up or `stop` is set."""
        self.nodes += 1
        if self.stop is not None and self.stop.is_set():
            raise SearchTimeout("stopped")
        if self.deadline is not None and time.perf_counter() > self.deadline:
            raise SearchTimeout("deadline passed")


def negamax(board: chess.Board, depth: int, evaluate: Evaluate, *,
            ctl: SearchControl | None = None) -> int:
    """The minimax value of `board` searched `depth` plies deep, side to move's view.

    No pruning: it visits every position. Used to check alphabeta and to count
    how many positions pruning saves.
    """
    if ctl is not None:
        ctl.visit()
    # Game over is checked before depth, so a mate on the horizon still scores as a mate.
    if board.is_game_over():
        if board.is_checkmate():
            return -(MATE_SCORE + depth)   # side to move is mated; sooner mates have more depth left
        return 0                           # stalemate, insufficient material, repetition, ...
    if depth == 0:
        return evaluate(board)

    best = -INFINITY
    for move in board.legal_moves:
        board.push(move)
        try:
            score = -negamax(board, depth - 1, evaluate, ctl=ctl)   # the child's score is from the opponent's view
        finally:
            board.pop()
        best = max(best, score)
    return best


def alphabeta(board: chess.Board, depth: int, alpha: int, beta: int, evaluate: Evaluate,
              quiesce: bool = False, *, ctl: SearchControl | None = None) -> int:
    """Negamax with alpha-beta pruning inside the window (alpha, beta).

    With the full window (-INFINITY, INFINITY) it must return exactly what
    negamax returns. With quiesce=True, call `quiescence` instead of
    `evaluate` at depth 0 (tutorial 4); the step tests use quiesce=False.

    `ctl` (passed down by root_search) counts nodes and raises SearchTimeout;
    the board is restored on the way out either way.
    """
    if ctl is not None:
        ctl.visit()
    if board.is_game_over():
        if board.is_checkmate():
            return -(MATE_SCORE + depth)
        return 0
    if depth == 0:
        return quiescence(board, alpha, beta, evaluate, ctl=ctl) if quiesce else evaluate(board)

    best = -INFINITY
    for move in order_moves(board):
        board.push(move)
        try:
            # The window is negated AND swapped: the child's (alpha, beta) is our (-beta, -alpha).
            score = -alphabeta(board, depth - 1, -beta, -alpha, evaluate, quiesce, ctl=ctl)
        finally:
            board.pop()   # also runs when SearchTimeout unwinds the recursion
        if score > best:
            best = score
        if score > alpha:
            alpha = score
        if alpha >= beta:
            break          # beta cutoff: the opponent will never allow this line
    return best            # fail-soft


def _victim_value(board: chess.Board, move: chess.Move) -> int:
    if board.is_en_passant(move):
        return PIECE_VALUES[chess.PAWN]     # the captured pawn is not on the destination square
    return PIECE_VALUES[board.piece_type_at(move.to_square)]


def _mvv_lva(board: chess.Board, move: chess.Move) -> tuple[int, int]:
    """Sort key for a capture: most valuable victim first, then least valuable attacker."""
    attacker = board.piece_type_at(move.from_square)
    # The king is worth 0 in PIECE_VALUES but is the riskiest attacker: try it last.
    attacker_value = INFINITY if attacker == chess.KING else PIECE_VALUES[attacker]
    return -_victim_value(board, move), attacker_value


def _ordered_captures(board: chess.Board) -> list[chess.Move]:
    return sorted(board.generate_legal_captures(), key=lambda m: _mvv_lva(board, m))


def order_moves(board: chess.Board) -> list[chess.Move]:
    """All legal moves, the most promising first (captures before quiet moves).

    Searching good moves first makes alpha-beta prune much more.
    Captures by MVV-LVA, then non-capturing promotions (queen first), then the rest.
    """
    captures = _ordered_captures(board)
    promotions: list[chess.Move] = []
    quiet: list[chess.Move] = []
    for move in board.legal_moves:
        if board.is_capture(move):
            continue
        (promotions if move.promotion else quiet).append(move)
    promotions.sort(key=lambda m: -PIECE_VALUES[m.promotion])
    return captures + promotions + quiet


def quiescence(board: chess.Board, alpha: int, beta: int, evaluate: Evaluate, *,
               ctl: SearchControl | None = None) -> int:
    """Keep searching captures only until the position is quiet.

    Stand pat on evaluate(board), then try captures only (board.is_capture).
    Do not test for checkmate or check here.
    """
    if ctl is not None:
        ctl.visit()
    stand_pat = evaluate(board)
    if stand_pat >= beta:
        return stand_pat
    alpha = max(alpha, stand_pat)
    for move in _ordered_captures(board):
        board.push(move)
        try:
            score = -quiescence(board, -beta, -alpha, evaluate, ctl=ctl)
        finally:
            board.pop()
        if score >= beta:
            return score
        alpha = max(alpha, score)
    return alpha


def root_search(board: chess.Board, depth: int, evaluate: Evaluate,
                deadline: float | None = None,
                stop: threading.Event | None = None,
                quiesce: bool = False, *,
                first: chess.Move | None = None,
                ctl: SearchControl | None = None) -> tuple[chess.Move, int]:
    """The best move at `depth` and its score (side to move's view); `quiesce` as in alphabeta.

    If `deadline` (a time.perf_counter() value) passes or `stop` is set, raise
    SearchTimeout so the engine can fall back to the last finished depth.

    `first` (keyword) is searched before every other move: iterative deepening
    passes the previous depth's best move. Pass your own `ctl` to read the
    node count afterwards; its deadline/stop then take the place of the arguments.
    """
    if ctl is None:
        ctl = SearchControl(deadline, stop)
    moves = order_moves(board)
    if first is not None and first in moves:
        moves.remove(first)
        moves.insert(0, first)

    ctl.visit()
    best_move: chess.Move | None = None
    best = -INFINITY
    alpha = -INFINITY
    for move in moves:
        board.push(move)
        try:
            score = -alphabeta(board, depth - 1, -INFINITY, -alpha, evaluate, quiesce, ctl=ctl)
        finally:
            board.pop()
        # Strictly better only: a later move that fails low returns exactly the
        # bound, ties with the best, and must not replace it.
        if score > best:
            best, best_move = score, move
        alpha = max(alpha, score)
    if best_move is None:
        raise ValueError("root_search: no legal moves (the game is over)")
    return best_move, best


def _deadline(limit: SearchLimit, started: float) -> float | None:
    return None if limit.time_s is None else started + limit.time_s


class MinimaxEngine:
    """Plain negamax to a fixed depth: the 'no pruning' baseline for the bench tool."""

    name: ClassVar[str] = "minimax"
    evaluation: ClassVar[str] = f"{EVAL_NAME} v{EVAL_VERSION}"

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng

    def search(self, board: chess.Board, limit: SearchLimit, stop: threading.Event) -> SearchResult:
        """Search to limit.depth (3 if not set) with negamax; report nodes, depth and score.

        If time runs out or `stop` is set first, it plays the best of the root
        moves searched so far (and reports no depth): this engine is a
        baseline for fixed-depth benchmarks, not for timed games.
        """
        depth = limit.depth or DEFAULT_DEPTH
        ctl = SearchControl(_deadline(limit, time.perf_counter()), stop)
        moves = list(board.legal_moves)
        best_move, best = moves[0], -INFINITY
        finished = False
        try:
            ctl.visit()
            for move in moves:
                board.push(move)
                try:
                    score = -negamax(board, depth - 1, evaluate, ctl=ctl)
                finally:
                    board.pop()
                if score > best:
                    best, best_move = score, move
            finished = True
        except SearchTimeout:
            pass
        info = SearchInfo(nodes=ctl.nodes, depth=depth if finished else None,
                          score_cp=best if finished else None, pv=[best_move])
        return SearchResult(best_move, info)


class AlphaBetaEngine:
    """Alpha-beta with iterative deepening: the team's main alpha-beta engine."""

    name: ClassVar[str] = "alphabeta"
    evaluation: ClassVar[str] = f"{EVAL_NAME} v{EVAL_VERSION}"

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng

    def search(self, board: chess.Board, limit: SearchLimit, stop: threading.Event) -> SearchResult:
        """Deepen 1, 2, 3, ... until limit.depth, limit.time_s or `stop`; play the last
        finished depth's move. Use quiesce=USE_QUIESCENCE and the shared `evaluate`.
        Report depth, nodes, score_cp and pv in SearchInfo."""
        started = time.perf_counter()
        if limit.depth is not None:
            max_depth = limit.depth
        elif limit.time_s is not None:
            max_depth = MAX_DEPTH
        else:
            max_depth = DEFAULT_DEPTH        # e.g. an MCTS-style iterations-only limit
        ctl = SearchControl(_deadline(limit, started), stop)

        best_move = order_moves(board)[0]    # always have something legal to play
        info = SearchInfo(pv=[best_move])
        for depth in range(1, max_depth + 1):
            try:
                move, score = root_search(board, depth, evaluate, quiesce=USE_QUIESCENCE,
                                          first=best_move if depth > 1 else None, ctl=ctl)
            except SearchTimeout:
                break                        # never play a half-searched depth's move
            best_move = move
            info = SearchInfo(depth=depth, score_cp=score, pv=[move])
            if abs(score) >= MATE_SCORE:
                break                        # a forced mate (for either side): deeper won't change the move
            if limit.time_s is not None and time.perf_counter() - started > limit.time_s / 2:
                break                        # the next depth costs several times this one: it won't finish
        info.nodes = ctl.nodes
        return SearchResult(best_move, info)
