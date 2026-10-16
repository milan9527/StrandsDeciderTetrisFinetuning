"""Real Strands Decider transport and bounded, two-stage placement selection."""

import asyncio
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from urllib.parse import urlsplit

import httpx

from backend.policy import POLICIES, dominates


class DeciderError(Exception):
    """A safe, user-visible error: never carries credentials or upstream response bodies."""


@dataclass(frozen=True)
class Settings:
    transport: str = "direct"
    url: str = "http://127.0.0.1:8099"
    credentials_file: str | None = None
    timeout: float = 30

    @classmethod
    def from_env(cls):
        settings = cls(
            os.getenv("DECIDER_TRANSPORT", "direct"),
            os.getenv("DECIDER_URL", "http://127.0.0.1:8099").rstrip("/"),
            os.getenv("DECIDER_CREDENTIALS_FILE") or None,
            float(os.getenv("DECIDER_TIMEOUT", "30")),
        )
        if settings.transport not in ("direct", "gateway"):
            raise ValueError("DECIDER_TRANSPORT must be direct or gateway")
        parsed = urlsplit(settings.url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username:
            raise ValueError("DECIDER_URL must be an HTTP(S) service URL without credentials")
        if not 1 <= settings.timeout <= 120:
            raise ValueError("DECIDER_TIMEOUT must be between 1 and 120 seconds")
        return settings


def valid_number(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def validate_result(result, questions, gateway=False, provider="decider"):
    if not isinstance(result, dict):
        raise DeciderError("模型返回了无效数据，棋盘未改变。")
    if gateway and result.get("source") != "live":
        raise DeciderError("服务未返回真实推理结果，已停止执行。")
    model = result.get("model")
    if not isinstance(model, str) or len(model) > 200 or not (
        provider == "decider" and "decider" in model.lower()
        or provider == "jev" and model.startswith("jev-")
        or provider == "gpt6" and model == "us.openai.gpt-6.1-sol"
    ):
        raise DeciderError("服务返回的模型与所选模型不符，已停止执行。")
    answers = result.get("answers")
    if not isinstance(answers, dict) or set(answers) != set(questions):
        raise DeciderError("模型回答不完整，棋盘未改变。")
    for name, question in questions.items():
        answer = answers[name]
        if not isinstance(answer, dict) or answer.get("type") != "choice":
            raise DeciderError("模型回答类型不正确，棋盘未改变。")
        if provider == "gpt6":
            if (
                not isinstance(answer.get("choice"), str)
                or answer["choice"] not in question["criteria"]
                or "probabilities" not in answer or answer["probabilities"] is not None
                or "confidence" not in answer or answer["confidence"] is not None
            ):
                raise DeciderError("GPT6 返回了无效落点或不受支持的概率，棋盘未改变。")
            continue
        probs = answer.get("probabilities")
        if (
            not isinstance(probs, dict)
            or set(probs) != set(question["criteria"])
            or not all(valid_number(p) for p in probs.values())
            or not math.isclose(sum(probs.values()), 1, abs_tol=0.02)
            or answer.get("choice") not in probs
            or not valid_number(answer.get("confidence"))
        ):
            raise DeciderError("模型概率或落点无效，棋盘未改变。")
        if probs[answer["choice"]] + 1e-6 < max(probs.values()):
            raise DeciderError("模型选择与返回的概率不一致，棋盘未改变。")


class DeciderClient:
    provider = "decider"
    configured = True
    model_id = "service-configured"

    def __init__(self, settings, client=None):
        self.settings = settings
        self.http = client or httpx.AsyncClient(
            base_url=settings.url,
            headers={"X-Decider-Client": "web"},
            timeout=httpx.Timeout(settings.timeout, connect=5),
            trust_env=False,
            follow_redirects=False,
        )
        self.authenticated = False
        self.auth_lock = asyncio.Lock()
        self.inference_lock = asyncio.Lock()

    async def _login(self):
        if self.settings.transport != "gateway" or self.authenticated:
            return
        async with self.auth_lock:
            if self.authenticated:
                return
            try:
                data = json.loads(Path(self.settings.credentials_file or "").read_text())
                credentials = {"username": data["username"], "password": data["password"]}
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise DeciderError("后端缺少有效的模型网关凭据，请检查服务配置。") from exc
            response = await self.http.post("/api/auth/login", json=credentials)
            if response.status_code != 200:
                raise DeciderError("模型网关登录失败，请检查后端配置的账号。")
            self.authenticated = True

    async def _request(self, method, path, **kwargs):
        try:
            await self._login()
            response = await self.http.request(method, path, **kwargs)
            if response.status_code == 401 and self.settings.transport == "gateway":
                self.authenticated = False
                await self._login()
                response = await self.http.request(method, path, **kwargs)
            if not response.is_success:
                raise DeciderError(
                    f"Decider 服务暂不可用（HTTP {response.status_code}），可恢复后重试。"
                )
            return response.json()
        except httpx.TimeoutException as exc:
            raise DeciderError("Decider 响应超时，已暂停。可以重试当前方块。") from exc
        except httpx.HTTPError as exc:
            raise DeciderError("无法连接 Decider，请检查模型服务后重试。") from exc
        except (ValueError, TypeError) as exc:
            raise DeciderError("Decider 返回格式无法读取，已暂停。") from exc

    async def health(self):
        try:
            data = await self._request(
                "GET",
                "/api/health" if self.settings.transport == "gateway" else "/health",
                timeout=6,
            )
            upstream = data.get("upstream", {}) if self.settings.transport == "gateway" else data
            connected = (
                data.get("connected")
                if self.settings.transport == "gateway"
                else data.get("status") == "ok"
            )
            model = upstream.get("model", "")
            if not connected or not isinstance(model, str) or "decider" not in model.lower():
                raise DeciderError("Decider 模型尚未就绪。")
            return {"connected": True, "model": model, "transport": self.settings.transport}
        except DeciderError as exc:
            return {"connected": False, "message": str(exc), "transport": self.settings.transport}

    async def evaluate(self, payload):
        # Shared GPU: admit one request at a time across this app's sessions.
        async with self.inference_lock:
            start = perf_counter()
            gateway = self.settings.transport == "gateway"
            result = await self._request(
                "POST",
                "/api/decide" if gateway else "/v1/systemone",
                json={"mode": "live", **payload} if gateway else payload,
            )
            validate_result(result, payload["questions"], gateway)
            return {
                "provider": self.provider,
                "payload": payload,
                "response": result,
                "elapsed_ms": round((perf_counter() - start) * 1000, 2),
            }

    async def close(self):
        try:
            if self.authenticated:
                await self.http.post("/api/auth/logout", json={}, timeout=5)
        except httpx.HTTPError:
            pass
        finally:
            await self.http.aclose()


LEGACY_INSTRUCTIONS = (
    "Play Tetris on a 10-column, 20-row board. Choose the best reachable landing "
    "from THESE candidates. Prefer a placement allowing the next piece to spawn. "
    "Avoid creating holes (empty cells below blocks); fewer holes is better. "
    "Clear lines when this does not create holes. Keep maximum height, total column "
    "heights and adjacent-height differences low to survive and clear future lines. "
    "All metrics describe the resulting board after line clears. "
    "Use the numerical candidate facts; option IDs do not indicate quality."
)

INSTRUCTIONS = (
    "Choose the Tetris move that clears rows and avoids buried holes. "
    "Fewer holes, lower stacks and less surface unevenness are better. "
    "A move that is worse in every measure should never be chosen."
)


def outcome_description(candidate):
    m = candidate.metrics
    change = m["holes_delta"]
    effect = (
        f"creates {change} new buried holes"
        if change > 0
        else f"removes {-change} buried holes" if change < 0 else "creates no new buried holes"
    )
    return (
        f"Clear {m['lines_cleared']} complete rows; {effect}; "
        f"leave {m['holes']} holes, a tallest stack of {m['max_height']} cells, "
        f"total height {m['aggregate_height']} and surface unevenness {m['bumpiness']}. "
        + ("The next piece can spawn." if m["next_spawn_ok"] else "The game ends immediately.")
    )


def question(candidates, policy="model"):
    return {
        "type": "choice",
        "instructions": LEGACY_INSTRUCTIONS if policy == "original" else INSTRUCTIONS,
        "criteria": {
            candidate.id: (
                candidate.description() if policy == "original" else outcome_description(candidate)
            )
            for candidate in candidates
        },
    }


def state_text(game):
    return json.dumps(
        {
            "game": "Tetris",
            "board_rows_top_to_bottom": [
                "".join("#" if v else "." for v in row) for row in game.board
            ],
            "coordinates": "x=0..9 left-to-right; y=0..19 top-to-bottom",
            "current_piece": game.active,
            "next_piece": game.queue[0],
            "lines_cleared_so_far": game.lines,
        },
        separators=(",", ":"),
    )


async def select_placement(
    game, candidates, client, progress, record=lambda receipt: None, policy="model"
):
    if policy not in POLICIES:
        raise ValueError("Unknown decision policy")
    selected, decision = await _select_placement(game, candidates, client, progress, record, policy)
    decision.update(
        policy=policy,
        eligible_count=len(candidates),
        candidate_count=len(candidates),
        exclusions=[],
        dominated_by=[p.id for p in candidates if dominates(p, selected)],
    )
    return selected, decision


async def _select_placement(game, candidates, client, progress, record, policy):
    if not candidates:
        raise DeciderError("没有可到达的合法落点。")
    if len(candidates) == 1:
        return candidates[0], {
            "source": "forced",
            "model": None,
            "calls": [],
            "elapsed_ms": 0,
            "confidence": None,
            "probabilities": None,
            "finalists": [candidates[0].public()],
            "candidate_count": 1,
            "group_count": 0,
        }
    started = perf_counter()
    # Shuffling depends only on the seed and piece count, never on a heuristic score.
    # Rotate ordering to avoid always grouping together the same orientations.
    import random

    ordered = list(candidates)
    random.Random(game.seed * 1009 + game.pieces).shuffle(ordered)
    groups = [ordered[i : i + 12] for i in range(0, len(ordered), 12)]
    calls = []
    winners = ordered
    state = state_text(game)
    if len(groups) > 1:
        winners = []
        # Gateway allows at most 8 questions per call.
        for offset in range(0, len(groups), 8):
            batch = groups[offset : offset + 8]
            questions = {
                f"group_{offset + n}": question(group, policy)
                for n, group in enumerate(batch)
                if len(group) > 1
            }
            progress(f"分组初选 · {len(candidates)} 个可达落点")
            answers = {}
            if questions:
                receipt = await client.evaluate({"state": state, "questions": questions})
                calls.append(receipt)
                record(receipt)
                answers = receipt["response"]["answers"]
            for n, group in enumerate(batch):
                chosen = answers[f"group_{offset + n}"]["choice"] if len(group) > 1 else group[0].id
                winners.append(next(c for c in group if c.id == chosen))
    # Rare complex boards can produce many reachable underhang landings; keep the same
    # bounded tournament until the final question fits the gateway's 12-option limit.
    while len(winners) > 12:
        next_winners = []
        for i in range(0, len(winners), 12):
            group = winners[i : i + 12]
            if len(group) == 1:
                next_winners.extend(group)
                continue
            progress("继续初选 · 复杂棋盘")
            receipt = await client.evaluate(
                {"state": state, "questions": {"advance": question(group, policy)}}
            )
            calls.append(receipt)
            record(receipt)
            chosen = receipt["response"]["answers"]["advance"]["choice"]
            next_winners.append(next(c for c in group if c.id == chosen))
        winners = next_winners
    progress(f"模型决选 · {len(winners)} 个候选")
    receipt = await client.evaluate(
        {"state": state, "questions": {"landing": question(winners, policy)}}
    )
    calls.append(receipt)
    record(receipt)
    answer = receipt["response"]["answers"]["landing"]
    selected = next(c for c in winners if c.id == answer["choice"])
    return selected, {
        "source": "live",
        "model": receipt["response"]["model"],
        "calls": calls,
        "elapsed_ms": round((perf_counter() - started) * 1000, 2),
        "confidence": answer["confidence"],
        "probabilities": answer["probabilities"],
        "finalists": [c.public() for c in winners],
        "candidate_count": len(candidates),
        "group_count": len(groups),
    }
