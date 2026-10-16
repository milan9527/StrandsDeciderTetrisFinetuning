"""In-memory games and cancellable-by-generation autonomous decision loops."""

import asyncio
import logging
import statistics
import uuid
from datetime import UTC, datetime
from time import monotonic

from backend.decider import DeciderError, select_placement
from backend.engine import Game
from backend.policy import POLICIES

logger = logging.getLogger(__name__)


class Session:
    def __init__(self, client, seed, max_pieces, interval_ms=500, idle_seconds=30, policy="model", provider="decider"):
        if policy not in POLICIES:
            raise ValueError("Unknown decision policy")
        self.id = uuid.uuid4().hex
        self.client = client
        self.policy = policy
        self.provider = provider
        self.game = Game(seed, max_pieces)
        self.candidates = self.game.candidates()
        self.interval_ms = interval_ms
        self.idle_seconds = idle_seconds
        self.last_seen = monotonic()
        self.created_at = datetime.now(UTC).isoformat()
        self.status = "ready"
        self.phase = "准备就绪"
        self.error = None
        self.epoch = 0
        self.revision = 0
        self.busy = False
        self.task = None
        self.history = []
        self.playback = []
        self.last_move = None
        self.last_decision = None
        self.request_latencies = []
        self.decision_latencies = []
        self.events = []

    def touch(self):
        self.last_seen = monotonic()

    def event(self, text, kind="info"):
        self.events.append(
            {
                "id": len(self.events) + 1,
                "time": datetime.now(UTC).isoformat(),
                "text": text,
                "kind": kind,
            }
        )
        self.revision += 1

    def start(self, single=False):
        if self.game.over:
            raise ValueError("本局已结束，请开始新的一局。")
        if self.busy or (self.task and not self.task.done()):
            raise ValueError("当前请求仍在结束中，请稍后再试。")
        self.touch()
        self.epoch += 1
        self.error = None
        self.status = "stepping" if single else "running"
        self.event("单步决策开始" if single else "模型开始自动游戏")
        self.task = asyncio.create_task(self._run(self.epoch, single))

    def pause(self, reason="已暂停"):
        self.epoch += 1
        if not self.game.over:
            self.status = "paused"
        self.phase = reason
        self.event(reason)

    async def _run(self, epoch, single):
        try:
            while epoch == self.epoch and not self.game.over:
                if monotonic() - self.last_seen > self.idle_seconds:
                    self.pause("页面已断开，自动暂停")
                    break
                if not self.candidates:
                    self.game.over, self.game.finish_reason = True, "top_out"
                    break
                self.busy = True
                receipts = []
                record = {
                    "turn": self.game.pieces + 1,
                    "piece": self.game.active,
                    "board_before": [row[:] for row in self.game.board],
                    "candidates": [c.public() for c in self.candidates],
                    "calls": receipts,
                    "started_at": datetime.now(UTC).isoformat(),
                }

                def progress(text):
                    if self.epoch == epoch:
                        self.phase = text
                        self.revision += 1

                def capture(receipt, receipts=receipts):
                    receipts.append(receipt)
                    self.request_latencies.append(receipt["elapsed_ms"])

                try:
                    chosen, decision = await select_placement(
                        self.game, self.candidates, self.client, progress, capture, policy=self.policy
                    )
                    if epoch != self.epoch or monotonic() - self.last_seen > self.idle_seconds:
                        record["outcome"] = "discarded_after_pause"
                        self.history.append(record)
                        if epoch == self.epoch:
                            self.pause("页面已断开，已丢弃晚到的决策")
                        break
                    self.last_move = self.game.apply(chosen)
                    self.last_decision = {k: v for k, v in decision.items() if k != "calls"}
                    self.last_decision["provider"] = self.provider
                    self.last_decision["selected"] = chosen.id
                    self.last_decision["selected_placement"] = chosen.public()
                    self.last_decision["request_count"] = len(receipts)
                    record.update(
                        outcome="applied",
                        decision=self.last_decision,
                        move=self.last_move,
                        board_after=[row[:] for row in self.game.board],
                        score=self.game.score,
                        lines=self.game.lines,
                    )
                    self.history.append(record)
                    # Keep every applied result for browsers whose polls span multiple turns.
                    # These are execution receipts, never another model/strategy decision.
                    self.playback.append({
                        "game": self.game.public(),
                        "move": self.last_move,
                        "decision": self.last_decision,
                    })
                    if decision["source"] == "live":
                        self.decision_latencies.append(decision["elapsed_ms"])
                    count = chosen.metrics["lines_cleared"]
                    suffix = f" · 消除 {count} 行" if count else ""
                    if decision["dominated_by"]:
                        suffix += " · 存在指标更优的落点"
                    self.event(
                        f"第 {self.game.pieces} 块 {chosen.piece} → 第 {chosen.x + 1} 列{suffix}",
                        "clear" if count else "move",
                    )
                    self.candidates = self.game.candidates()
                    self.phase = "落点已执行"
                except DeciderError as exc:
                    record.update(outcome="model_error", message=str(exc))
                    self.history.append(record)
                    if epoch == self.epoch:
                        self.error = str(exc)
                        self.status = "error"
                        self.phase = "模型调用失败"
                        self.event(str(exc), "error")
                    break
                finally:
                    self.busy = False
                    self.revision += 1
                if single:
                    if not self.game.over:
                        self.status, self.phase = "paused", "单步完成"
                    break
                await asyncio.sleep(self.interval_ms / 1000)
            if self.game.over:
                self.status = "finished"
                self.phase = (
                    "达到方块上限" if self.game.finish_reason == "piece_limit" else "棋盘已满"
                )
                self.event(f"本局结束 · {self.phase}", "finish")
        except asyncio.CancelledError:
            self.busy = False
            raise
        except Exception:
            logger.exception("Game loop failed for session %s", self.id)
            self.busy = False
            self.status = "error"
            self.phase = "游戏已暂停"
            self.error = "游戏运行异常，棋盘已保留。请新建一局。"
            self.event(self.error, "error")

    def snapshot(self, after_piece=None):
        latencies = self.decision_latencies
        return {
            "id": self.id,
            "revision": self.revision,
            "created_at": self.created_at,
            "status": self.status,
            "phase": self.phase,
            "busy": self.busy,
            "error": self.error,
            "interval_ms": self.interval_ms,
            "policy": self.policy,
            "provider": self.provider,
            "requested_model": getattr(self.client, "model_id", None),
            "game": self.game.public(),
            "candidate_count": len(self.candidates),
            "candidates": [p.public() for p in self.candidates],
            "last_move": self.last_move,
            "decision": self.last_decision,
            "stats": {
                "successful_requests": len(self.request_latencies),
                "live_moves": len(latencies),
                "forced_moves": sum(
                    h.get("decision", {}).get("source") == "forced" for h in self.history
                ),
                "policy_moves": 0,  # Retained for export compatibility; no rule strategy runs.
                "dominated_moves": sum(
                    bool(h.get("decision", {}).get("dominated_by")) for h in self.history
                ),
                "errors": sum(h["outcome"] == "model_error" for h in self.history),
                "mean_decision_ms": round(statistics.mean(latencies), 1) if latencies else None,
                "last_decision_ms": latencies[-1] if latencies else None,
                "p95_decision_ms": (
                    sorted(latencies)[max(0, (95 * len(latencies) + 99) // 100 - 1)]
                    if latencies
                    else None
                ),
            },
            "events": self.events[-30:],
            **({"playback": [
                frame for frame in self.playback
                if frame["game"]["pieces"] > after_piece
            ][:100]} if after_piece is not None else {}),
        }

    async def close(self):
        self.pause("本局已关闭")
        if self.task and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass


class SessionStore:
    def __init__(self, client, maximum=8, idle_seconds=30, providers=None):
        self.client = client
        self.providers = providers or {"decider": client}
        self.maximum = maximum
        self.idle_seconds = idle_seconds
        self.sessions = {}

    async def create(self, seed, max_pieces, interval_ms, policy="model", provider="decider"):
        client = self.providers.get(provider)
        if client is None or not getattr(client, "configured", True):
            raise DeciderError("所选模型尚未配置，请检查后端模型设置。")
        # Free abandoned games. Reading the export/state is a heartbeat.
        for key, old in list(self.sessions.items()):
            if monotonic() - old.last_seen > 600:
                await old.close()
                del self.sessions[key]
        if len(self.sessions) >= self.maximum:
            raise ValueError("当前游戏数量已达上限，请关闭旧局后重试。")
        session = Session(client, seed, max_pieces, interval_ms, self.idle_seconds, policy, provider)
        self.sessions[session.id] = session
        return session

    async def close(self):
        for session in self.sessions.values():
            await session.close()
