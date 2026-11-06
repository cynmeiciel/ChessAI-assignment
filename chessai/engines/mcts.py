"""MCTS with progressive widening, capture-ordered expansion and uniform rollouts.

Conventions (the tests rely on them):
- Values are rewards in [0, 1]: 1 = win, 0.5 = draw, 0 = loss.
  Nonterminal rewards are heuristic estimates, not calibrated probabilities.
- `Node.value_sum` is from the point of view of the player who MADE
  `node.move` (the player to move at the parent). A parent therefore picks
  the child with the best value_sum / visits for itself.
- `rollout` and `backpropagate` take and return values from the point of
  view of the SIDE TO MOVE in that node's position.
"""
from __future__ import annotations

import math
import json
import numbers
import os
import random
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar
from uuid import uuid4

import chess

from chessai.engines.base import Candidate, SearchInfo, SearchLimit, SearchResult
from chessai.engines.evaluate import EVAL_NAME, EVAL_VERSION, PIECE_VALUES, evaluate

EXPLORATION = 0.5
ROLLOUT_PLIES = 2

Evaluate = Callable[[chess.Board], int]
_LOG_LOCK = threading.Lock()


class _SearchLog:
    """Append structured diagnostics outside the repository; never print in the UI.

    MCTS_LOG_FILE overrides the path; 'off' disables tracing. Logging failures
    are reported on engine.last_log_error and must not prevent legal play.
    """

    def __init__(self) -> None:
        configured = os.environ.get("MCTS_LOG_FILE")
        self.path = (Path(configured).expanduser() if configured is not None
                     else Path(__file__).resolve().parents[3] / "mcts-search.jsonl")
        self.enabled = configured is None or configured.strip().lower() not in ("", "off", "0", "false")
        self.search_id = uuid4().hex
        self.error: str | None = None
        self.file = None
        if self.enabled:
            try:
                self.path = self.path.resolve()
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.file = self.path.open("a", encoding="utf-8", buffering=1)
            except OSError as exc:
                self.error = str(exc)
                self.enabled = False

    def emit(self, event: str, **data) -> None:
        if self.file is None or self.error is not None:
            return
        try:
            line = json.dumps({"event": event, "search_id": self.search_id, **data}, ensure_ascii=False)
            with _LOG_LOCK:
                self.file.write(line + "\n")
        except (OSError, TypeError, ValueError) as exc:
            self.error = str(exc)

    def close(self) -> None:
        if self.file is not None:
            try:
                self.file.close()
            except OSError as exc:
                self.error = str(exc)


class _SearchInterrupted(Exception):
    """Internal control flow: an unfinished iteration contributes no reward."""


def _real(name: str, value: float, *, zero: bool = False) -> None:
    if (isinstance(value, bool) or not isinstance(value, numbers.Real)
            or not math.isfinite(value) or value < 0 or (not zero and value == 0)):
        raise ValueError(f"{name} must be a finite {'nonnegative' if zero else 'positive'} number")


def _integer(name: str, value: int, *, zero: bool = False) -> None:
    if (isinstance(value, bool) or not isinstance(value, int)
            or value < 0 or (not zero and value == 0)):
        raise ValueError(f"{name} must be a {'nonnegative' if zero else 'positive'} integer")


@dataclass(frozen=True)
class MCTSConfig:
    """Fallback budgets apply only when SearchLimit supplies neither time nor iterations."""

    exploration_constant: float = EXPLORATION
    budget_mode: str = "simulations"
    max_simulations: int = 1000
    time_limit_ms: float = 1000
    rollout_max_plies: int = ROLLOUT_PLIES
    evaluation_scale: float = 400
    rollout_policy: str = "random"
    rollout_epsilon: float = 0.15  # accepted for API compatibility; no longer used
    reuse_tree: bool = True
    progressive_widening: bool = True
    widening_constant: float = 2.0
    widening_exponent: float = 0.5

    def __post_init__(self) -> None:
        _real("exploration_constant", self.exploration_constant, zero=True)
        _integer("max_simulations", self.max_simulations)
        _real("time_limit_ms", self.time_limit_ms)
        _integer("rollout_max_plies", self.rollout_max_plies, zero=True)
        _real("evaluation_scale", self.evaluation_scale)
        if self.budget_mode not in ("simulations", "time"):
            raise ValueError("budget_mode must be simulations or time")
        if self.rollout_policy not in ("random", "heuristic"):
            raise ValueError("rollout_policy must be random or heuristic")
        # Accept old configuration files without restoring heuristic rollouts.
        object.__setattr__(self, "rollout_policy", "random")
        _real("rollout_epsilon", self.rollout_epsilon, zero=True)
        if self.rollout_epsilon > 1:
            raise ValueError("rollout_epsilon must be in [0, 1]")
        if not isinstance(self.reuse_tree, bool):
            raise ValueError("reuse_tree must be a boolean")
        if not isinstance(self.progressive_widening, bool):
            raise ValueError("progressive_widening must be a boolean")
        _real("widening_constant", self.widening_constant)
        _real("widening_exponent", self.widening_exponent)
        if self.widening_exponent >= 1:
            raise ValueError("widening_exponent must be in (0, 1)")


@dataclass(frozen=True)
class MCTSStats:
    """Per-request diagnostics; rewards are heuristic estimates, not calibrated odds."""

    simulations: int = 0
    nodes_created: int = 0
    elapsed_ms: float = 0.0
    max_tree_depth: int = 0
    root_win_rate: float | None = None
    reused_tree: bool = False
    reused_visits: int = 0
    reused_nodes: int = 0
    root_visits: int = 0


@dataclass(eq=False)
class Node:
    """One position in the search tree. The position itself is not stored:
    replay the moves from the root (select/expand push them on a board)."""

    move: chess.Move | None              # the move that led here; None at the root
    parent: Node | None
    untried: list[chess.Move]            # legal moves with no child yet
    children: list[Node] = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0               # from the view of the player who made `move`

    @classmethod
    def root(cls, board: chess.Board) -> Node:
        return cls(None, None, [] if board.is_game_over() else list(board.legal_moves))


def uct_score(value_sum: float, visits: int, parent_visits: int, c: float) -> float:
    """UCT: average value plus an exploration bonus. math.inf for an unvisited child."""
    if visits == 0:
        return math.inf
    return value_sum / visits + c * math.sqrt(math.log(max(1, parent_visits)) / visits)


def _widening_limit(node: Node, widening: tuple[float, float]) -> int:
    """Grow the permitted width with visits; never discard an untried move."""
    k, alpha = widening
    total = len(node.children) + len(node.untried)
    return max(1, math.ceil(min(total, k * max(1, node.visits) ** alpha)))


def select(node: Node, board: chess.Board, c: float,
           *, _check: Callable[[], None] | None = None,
           _trace: list[dict] | None = None,
           _widening: tuple[float, float] | None = None) -> Node:
    """Follow standard UCT until terminal or another expansion is permitted.

    By default helpers retain classic full-expansion behavior. With _widening,
    follow existing children once the width cap is reached, even if untried
    moves remain. Push every traversed move and return the reached node.
    """
    while node.children and not board.is_game_over():
        if _check is not None:
            _check()
        if node.untried and (_widening is None or len(node.children) < _widening_limit(node, _widening)):
            break
        parent = node
        node = max(parent.children, key=lambda child: uct_score(
            child.value_sum, child.visits, node.visits, c))
        if _trace is not None:
            _trace.append(dict(move=node.move.uci(), parent_visits=parent.visits,
                               parent_children=len(parent.children), parent_untried=len(parent.untried),
                               widening_limit=_widening_limit(parent, _widening) if _widening else None,
                               visits=node.visits,
                               mean_reward=node.value_sum / node.visits if node.visits else None,
                               uct=uct_score(node.value_sum, node.visits, parent.visits, c)))
        board.push(node.move)
    if _check is not None:
        _check()
    return node


def expand(node: Node, board: chess.Board, rng: random.Random, *,
           _policy: str = "random", _preferred: chess.Move | None = None) -> Node:
    """Expand highest-value untried captures first, then all remaining moves.

    Equal-priority choices are uniform random. _policy is a legacy argument;
    it cannot enable heuristic selection. No legal move is pruned.
    """
    if not node.untried or board.is_game_over():
        return node
    captures = [move for move in node.untried if board.is_capture(move)]
    if captures:
        def victim_value(move: chess.Move) -> int:
            if board.is_en_passant(move):
                return PIECE_VALUES[chess.PAWN]
            return PIECE_VALUES[board.piece_type_at(move.to_square)]
        highest = max(victim_value(move) for move in captures)
        choices = [move for move in captures if victim_value(move) == highest]
    else:
        choices = node.untried
    if _preferred is not None and _preferred in choices:
        move = _preferred
    else:
        move = choices[rng.randrange(len(choices))]
    node.untried.remove(move)
    board.push(move)
    child = Node(move, node, [] if board.is_game_over() else list(board.legal_moves))
    node.children.append(child)
    return child


def cp_to_win(cp: int) -> float:
    """Turn a centipawn score into a win probability: 1 / (1 + 10 ** (-cp / 400))."""
    return _cp_to_win(cp, 400)


def _cp_to_win(cp: int, scale: float) -> float:
    # Exponent is never positive, so even a very large material score is safe.
    small = 10 ** (-abs(cp) / scale)
    return 1 / (1 + small) if cp >= 0 else small / (1 + small)


def _rollout_move(board: chess.Board, moves: list[chess.Move], rng: random.Random,
                  *, epsilon: float) -> chess.Move:
    """Legacy call form; uniformly sample supplied legal moves, ignoring epsilon."""
    return rng.choice(moves)


def rollout(board: chess.Board, rng: random.Random, max_plies: int, evaluate: Evaluate,
            *, evaluation_scale: float = 400,
            policy: str = "random", epsilon: float = 0.15,
            _check: Callable[[], None] | None = None,
            _trace: dict | None = None) -> float:
    """Play up to `max_plies` moves (chosen with `rng`) from `board`, then score it.

    Returns the value for the side to move in `board` AT THE START: the real
    result if the game ends, otherwise cp_to_win(evaluate(...)) converted to
    that side's view. Leave `board` as you found it.
    """
    _integer("max_plies", max_plies, zero=True)
    _real("evaluation_scale", evaluation_scale)
    if policy not in ("random", "heuristic"):
        raise ValueError("policy must be random or heuristic")
    policy = "random"  # historical 'heuristic' is accepted as an alias only
    _real("epsilon", epsilon, zero=True)
    if epsilon > 1:
        raise ValueError("epsilon must be in [0, 1]")
    start_side, start_depth = board.turn, len(board.move_stack)
    if _trace is not None:
        _trace.update(side="white" if start_side else "black", policy=policy, moves=[])
    try:
        for ply in range(max_plies + 1):
            if _check is not None:
                _check()
            outcome = board.outcome(claim_draw=False)
            if outcome is not None:
                if _trace is not None:
                    _trace.update(stop_reason="terminal", result=outcome.result(),
                                  termination=outcome.termination.name, evaluation_cp=None)
                return .5 if outcome.winner is None else float(outcome.winner == start_side)
            if ply == max_plies:
                cp = evaluate(board)
                value = _cp_to_win(cp, evaluation_scale)
                if _trace is not None:
                    _trace.update(stop_reason="ply_limit", evaluation_cp=cp,
                                  evaluation_side="white" if board.turn else "black")
                return value if board.turn == start_side else 1 - value
            moves = list(board.legal_moves)
            move = rng.choice(moves)
            board.push(move)
            if _trace is not None:
                _trace["moves"].append(move.uci())
    finally:
        if _trace is not None:
            _trace.setdefault("stop_reason", "interrupted_or_error")
            _trace["final_fen"] = board.fen()
        while len(board.move_stack) > start_depth:
            board.pop()


def backpropagate(node: Node, value: float) -> None:
    """Add one visit and the result to `node` and every ancestor up to the root.

    `value` is for the side to move at `node`'s position; remember that
    value_sum is stored for the player who moved INTO each node.
    """
    while node is not None:
        value = 1 - value
        node.visits += 1
        node.value_sum += value
        node = node.parent


def best_move(root: Node) -> chess.Move:
    """The move to play: the root child with the most visits."""
    visited = [child for child in root.children if child.visits]
    if visited:
        return max(visited, key=lambda child: child.visits).move
    if root.untried:
        return root.untried[0]
    if root.children:
        return root.children[0].move
    raise ValueError("no move available")


class MCTSEngine:
    """Retain matching subtrees; use a separate engine/RNG for concurrent requests.

    The shared material+PST evaluator keeps existing match metadata accurate.
    Additional per-request measurements are available in immutable last_stats.
    """

    name: ClassVar[str] = "mcts"
    evaluation: ClassVar[str] = f"{EVAL_NAME} v{EVAL_VERSION}"

    def __init__(self, rng: random.Random, *, config: MCTSConfig | None = None) -> None:
        self.rng = rng
        self.config = config if config is not None else MCTSConfig()
        self.last_stats = MCTSStats()
        self.last_log_path: str | None = None
        self.last_log_error: str | None = None
        self._root: Node | None = None
        self._root_board: chess.Board | None = None
        self._root_config: MCTSConfig | None = None

    def reset_tree(self) -> None:
        """Discard retained work, e.g. for a new independent experiment."""
        self._root = None
        self._root_board = None
        self._root_config = None

    def _prepare_root(self, board: chess.Board) -> tuple[Node, str, int, int]:
        """Reuse only an exact history continuation, never a FEN transposition."""
        root = None
        reason = "no_tree"
        old = self._root_board
        if not self.config.reuse_tree:
            reason = "disabled"
        elif self._root is not None and old is not None:
            history = board.move_stack
            old_history = old.move_stack
            if self._root_config != self.config:
                reason = "config_changed"
            elif (type(board) is not type(old) or board.chess960 != old.chess960
                  or board.root().fen(en_passant="fen", promoted=True)
                  != old.root().fen(en_passant="fen", promoted=True)
                  or len(history) < len(old_history)
                  or history[:len(old_history)] != old_history):
                reason = "history_mismatch"
            else:
                # Same moves can still arrive with manually changed clocks,
                # rights or en-passant state. Compare the exact anchor position.
                anchor = board.copy(stack=True)
                continuation = history[len(old_history):]
                for _ in continuation:
                    anchor.pop()
                if anchor.fen(en_passant="fen", promoted=True) != old.fen(en_passant="fen", promoted=True):
                    reason = "state_mismatch"
                else:
                    for move in continuation:
                        anchor.push(move)
                    if anchor.fen(en_passant="fen", promoted=True) != board.fen(en_passant="fen", promoted=True):
                        reason = "state_mismatch"
                    else:
                        root = self._root
                        for move in continuation:
                            root = next((child for child in root.children if child.move == move), None)
                            if root is None:
                                break
                        reason = ("descendant" if continuation else "same_position") if root is not None else "branch_not_found"
        if root is None:
            root = Node.root(board)
            reused_nodes = retained_depth = 0
        else:
            # The incoming mover is still the opponent of the current side;
            # neither visit counts nor reward sums need a perspective flip.
            root.parent = None
            root.move = None
            reused_nodes = retained_depth = 0
            stack = [(root, 0)]
            while stack:
                current, depth = stack.pop()
                reused_nodes += 1
                retained_depth = max(retained_depth, depth)
                stack.extend((child, depth + 1) for child in current.children)
        if self.config.reuse_tree:
            self._root = root
            self._root_board = board.copy(stack=True)
            self._root_config = self.config
        else:
            self.reset_tree()
        return root, reason, reused_nodes, retained_depth

    def search(self, board: chess.Board, limit: SearchLimit, stop: threading.Event) -> SearchResult:
        """Run select / expand / rollout / backpropagate until limit.iterations,
        limit.time_s or `stop`. Report info.nodes = iterations run and one
        Candidate(move, visits, value) per root child (value from the root
        mover's view). Incomplete interrupted iterations contribute no visits.

        The supplied board and its history are preserved. A stopped search has
        a legal fallback; an automatically terminal board raises ValueError.
        Depth-only requests use the finite configured fallback budget.
        Matching cached visits do not consume this call's budget. Candidate
        visits are new work for the UI; their values and order use all retained
        samples. Logs additionally expose total_visits and reuse diagnostics.
        """
        started = time.perf_counter()
        self.last_stats = MCTSStats()
        log = _SearchLog()
        self.last_log_path = str(log.path) if log.enabled else None
        self.last_log_error = None
        result = None
        status = "error"
        try:
            log.emit("search_started", timestamp_utc=datetime.now(timezone.utc).isoformat(),
                     fen=board.fen(), history=[move.uci() for move in board.move_stack],
                     side="white" if board.turn else "black", config=asdict(self.config),
                     limits=dict(time_s=limit.time_s, iterations=limit.iterations, depth=limit.depth),
                     evaluation=self.evaluation)
            result = self._search(board, limit, stop, started, log if log.enabled else None)
            status = "ok"
            return result
        except Exception as exc:
            self.reset_tree()
            log.emit("search_error", error_type=type(exc).__name__, message=str(exc))
            raise
        finally:
            self.last_stats = replace(self.last_stats, elapsed_ms=(time.perf_counter() - started) * 1000)
            total_visits = {child.move: child.visits for child in self._root.children} if self._root else {}
            log.emit("search_finished", status=status, stats=asdict(self.last_stats),
                     chosen_move=result.move.uci() if result else None,
                     chosen_san=board.san(result.move) if result else None,
                     selection_rule="most_visits" if result and result.info.candidates else "legal_fallback",
                     candidates=[dict(move=c.move.uci(), visits=c.visits,
                                      total_visits=total_visits.get(c.move, c.visits), mean_reward=c.value)
                                 for c in result.info.candidates] if result else [])
            log.close()
            self.last_log_error = log.error

    def _search(self, board: chess.Board, limit: SearchLimit,
                stop: threading.Event, started: float, log: _SearchLog | None = None) -> SearchResult:
        iterations, time_s = limit.iterations, limit.time_s
        if iterations is not None:
            _integer("iterations", iterations)
        if time_s is not None:
            _real("time_s", time_s)
        if iterations is None and time_s is None:
            if self.config.budget_mode == "time":
                time_s = self.config.time_limit_ms / 1000
            else:
                iterations = self.config.max_simulations
        deadline = None if time_s is None else started + time_s

        def check() -> None:
            if stop.is_set() or (deadline is not None and time.perf_counter() >= deadline):
                raise _SearchInterrupted

        if board.is_game_over():
            raise ValueError("game is already over")
        work = board.copy(stack=True)
        start_depth = len(work.move_stack)
        root, reuse_reason, reused_nodes, deepest = self._prepare_root(work)
        initial_visits = root.visits
        prior_child_visits = {child.move: child.visits for child in root.children}
        nodes_created = 0 if reused_nodes else 1
        widening = ((self.config.widening_constant, self.config.widening_exponent)
                    if self.config.progressive_widening else None)
        if log is not None:
            log.emit("tree_prepared", reused_tree=bool(reused_nodes), reuse_reason=reuse_reason,
                     reused_nodes=reused_nodes, reused_visits=initial_visits,
                     retained_depth=deepest)
        while iterations is None or root.visits - initial_visits < iterations:
            added = None
            selection = [] if log is not None else None
            rollout_trace = {} if log is not None else None
            tree_path = []
            try:
                check()
                parent = select(root, work, self.config.exploration_constant,
                                _check=check, _trace=selection, _widening=widening)
                check()
                node = expand(parent, work, self.rng)
                if node is not parent:
                    added = node
                    nodes_created += 1
                depth = len(work.move_stack) - start_depth
                if log is not None:
                    tree_path = [move.uci() for move in work.move_stack[start_depth:]]
                value = rollout(work, self.rng, self.config.rollout_max_plies, evaluate,
                                evaluation_scale=self.config.evaluation_scale,
                                policy=self.config.rollout_policy, epsilon=self.config.rollout_epsilon,
                                _check=check, _trace=rollout_trace)
                check()
                backpropagate(node, value)
                deepest = max(deepest, depth)
                if log is not None:
                    updates = []
                    current = node
                    while current is not None:
                        updates.append(dict(move=current.move.uci() if current.move else None,
                                            visits=current.visits, value_sum=current.value_sum))
                        current = current.parent
                    log.emit("iteration_completed", iteration=root.visits - initial_visits, selection=selection,
                             expanded_move=added.move.uci() if added else None, tree_path=tree_path,
                             tree_depth=depth, rollout=rollout_trace, reward_for_rollout_side=value,
                             backpropagation=updates, root_visits=root.visits)
            except _SearchInterrupted:
                if added is not None:
                    added.parent.children.remove(added)
                    added.parent.untried.append(added.move)
                if log is not None:
                    log.emit("iteration_aborted", next_iteration=root.visits - initial_visits + 1,
                             reason="cancelled" if stop.is_set() else "time_limit",
                             selection=selection, tree_path=tree_path, rollout=rollout_trace,
                             root_visits=root.visits)
                break
            finally:
                while len(work.move_stack) > start_depth:
                    work.pop()
        # SearchInfo is per request (the UI divides visits by info.nodes).
        # Selection and ordering still use the actual cumulative tree counts.
        candidates = [Candidate(child.move, child.visits - prior_child_visits.get(child.move, 0),
                                child.value_sum / child.visits)
                      for child in sorted(root.children, key=lambda c: c.visits, reverse=True)
                      if child.visits]
        self.last_stats = MCTSStats(
            simulations=root.visits - initial_visits, nodes_created=nodes_created, max_tree_depth=deepest,
            root_win_rate=1 - root.value_sum / root.visits if root.visits else None,
            reused_tree=bool(reused_nodes), reused_visits=initial_visits,
            reused_nodes=reused_nodes, root_visits=root.visits)
        return SearchResult(best_move(root), SearchInfo(nodes=root.visits - initial_visits, depth=deepest,
                                                       candidates=candidates))
