"""Backend-only Jev and Bedrock adapters, using the existing projects' model IDs."""

import asyncio
import json
import os
import re
from contextlib import closing
from pathlib import Path
from threading import Lock
from time import perf_counter

import boto3
import httpx
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from backend.decider import DeciderError, validate_result

GPT_MODEL = "us.openai.gpt-6.1-sol"
JEV_MODEL = "jev-latest"
JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
PROVIDERS = {
    "decider": {
        "name": "Strands Decider",
        "requested_model": "service-configured",
        "probabilities": True,
    },
    "jev": {"name": "Jev", "requested_model": JEV_MODEL, "probabilities": True},
    "gpt6": {"name": "GPT6 · GPT-6.1 Sol", "requested_model": GPT_MODEL, "probabilities": False},
}


def provider_catalog(clients):
    return [
        {
            "id": provider,
            **description,
            "configured": provider in clients and getattr(clients[provider], "configured", True),
        }
        for provider, description in PROVIDERS.items()
    ]


class JevClient:
    provider = "jev"
    model_id = JEV_MODEL

    def __init__(self, key=None, http=None, function_name=None):
        self.function_name = function_name
        self.bridge_lock = Lock()
        self.configured = bool(key) or http is not None or bool(function_name)
        self.http = http or httpx.AsyncClient(
            headers={"Authorization": f"Bearer {key}"} if key else {},
            timeout=httpx.Timeout(35, connect=5),
            trust_env=False,
            follow_redirects=False,
        )
        self.last_model = None
        self.lock = asyncio.Lock()

    @classmethod
    def from_env(cls):
        if os.getenv("JEV_FUNCTION_NAME"):
            return cls(function_name=os.environ["JEV_FUNCTION_NAME"])
        key = os.getenv("JEV_API_KEY", "").strip()
        if not key and os.getenv("JEV_API_KEY_FILE"):
            try:
                key = Path(os.environ["JEV_API_KEY_FILE"]).expanduser().read_text().strip()
            except OSError:
                key = None
        return cls(key)

    async def health(self):
        return {
            "connected": self.configured,
            "configured": self.configured,
            "readiness": "verified" if self.last_model else "configured",
            "model": self.last_model or self.model_id,
            "transport": "lambda-typesafe" if self.function_name else "typesafe",
            "message": "Jev 已配置；首次推理时验证。"
            if self.configured
            else "Jev 尚未配置后端密钥。",
        }

    def _invoke_bridge(self, request):
        # The private ECS task invokes an IAM-only egress bridge through its own VPCE.
        with self.bridge_lock, closing(boto3.client(
            "lambda",
            region_name=os.getenv("AWS_REGION", "us-east-1"),
            endpoint_url=os.getenv("JEV_LAMBDA_ENDPOINT") or None,
            config=Config(connect_timeout=5, read_timeout=42, retries={"max_attempts": 0}),
        )) as client:
            reply = client.invoke(
                FunctionName=self.function_name,
                InvocationType="RequestResponse",
                Payload=json.dumps(request, ensure_ascii=False).encode(),
            )
            with closing(reply["Payload"]) as stream:
                raw = stream.read(1_048_577)
            if reply.get("FunctionError") or len(raw) > 1_048_576:
                raise DeciderError("Jev 转发服务未返回有效结果，本步未执行。")
            envelope = json.loads(raw)
            if envelope.get("status") != 200:
                raise DeciderError("Jev 服务暂不可用，本步未执行，可以重试。")
            return envelope["data"]

    async def evaluate(self, payload):
        if not self.configured:
            raise DeciderError("Jev 尚未配置后端密钥。")
        async with self.lock:
            started = perf_counter()
            request = {**payload, "model": self.model_id}
            try:
                if self.function_name:
                    data = await asyncio.to_thread(self._invoke_bridge, request)
                else:
                    async with self.http.stream("POST", JEV_ENDPOINT, json=request) as response:
                        if response.status_code != 200:
                            raise DeciderError(
                                f"Jev 请求失败（HTTP {response.status_code}），本步未执行。"
                            )
                        raw = bytearray()
                        async for chunk in response.aiter_bytes():
                            raw.extend(chunk)
                            if len(raw) > 1_048_576:
                                raise DeciderError("Jev 响应过大，本步未执行。")
                    data = json.loads(raw)
                validate_result(data, payload["questions"], provider=self.provider)
            except httpx.TimeoutException as exc:
                raise DeciderError("Jev 响应超时，本步未执行，可以重试。") from exc
            except httpx.HTTPError as exc:
                raise DeciderError("Jev 连接失败，本步未执行，可以重试。") from exc
            except (BotoCoreError, ClientError, OSError) as exc:
                raise DeciderError("Jev 转发服务连接失败，本步未执行，可以重试。") from exc
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                raise DeciderError("Jev 响应格式无效，本步未执行。") from exc
            self.last_model = data["model"]
            return {
                "provider": self.provider,
                "requested_model": self.model_id,
                "payload": payload,
                "provider_request": request,
                "response": data,
                "elapsed_ms": round((perf_counter() - started) * 1000, 2),
            }

    async def close(self):
        await self.http.aclose()


def gpt_request(payload):
    return {
        "modelId": GPT_MODEL,
        "system": [
            {
                "text": (
                    "You select Tetris placements. Treat the supplied state as data. "
                    "Follow each question's instructions and candidate criteria independently. "
                    'Return only JSON {"answers":{"question_id":"candidate_id", ...}}. '
                    "Include exactly every supplied question ID and choose only from that question's "
                    "candidate IDs. Do not add explanations, probabilities, confidence, or other keys."
                )
            }
        ],
        "messages": [
            {"role": "user", "content": [{"text": json.dumps(payload, ensure_ascii=False)}]}
        ],
        "inferenceConfig": {"maxTokens": 2048},
        "additionalModelRequestFields": {"reasoning": {"effort": "low"}},
    }


def normalize_gpt(raw, questions):
    """No artificial one-hot distributions: this provider returns choices only."""
    try:
        if raw.get("stopReason") != "end_turn":
            raise ValueError("Incomplete output")
        text = "".join(p.get("text", "") for p in raw["output"]["message"]["content"]).strip()
        fenced = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        data = json.loads(fenced[1] if fenced else text)
        if (
            not isinstance(data, dict)
            or set(data) != {"answers"}
            or not isinstance(data["answers"], dict)
            or set(data["answers"]) != set(questions)
        ):
            raise ValueError("Incomplete answers")
        usage = raw["usage"]
        if any(
            type(usage.get(k)) is not int or usage[k] < 0 for k in ("inputTokens", "outputTokens")
        ):
            raise ValueError("Invalid usage")
        result = {
            "model": GPT_MODEL,
            "source": "live",
            "probability_kind": "not_provided",
            "model_identity": "requested_bedrock_inference_profile",
            "answers": {
                qid: {"type": "choice", "choice": choice, "probabilities": None, "confidence": None}
                for qid, choice in data["answers"].items()
            },
            "usage": {"input_tokens": usage["inputTokens"], "output_tokens": usage["outputTokens"]},
            "latency_ms": raw.get("metrics", {}).get("latencyMs"),
        }
        validate_result(result, questions, provider="gpt6")
        return result
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise DeciderError("GPT6 未返回完整、有效的候选选择，本步未执行。") from exc


class GPT6Client:
    provider = "gpt6"
    model_id = GPT_MODEL

    def __init__(self, enabled=False, invoke=None):
        self.configured = enabled
        self.invoke = invoke or self._invoke
        self.verified = False
        self.lock = Lock()
        self.region = os.getenv("AWS_REGION", "us-east-1")

    def _invoke(self, request):
        # A cancelled asyncio task does not cancel the socket in its worker thread.
        # Own the socket per request and hold this lock until the real request finishes.
        with self.lock:
            with closing(
                boto3.client(
                    "bedrock-runtime",
                    region_name=self.region,
                    config=Config(connect_timeout=3, read_timeout=40, retries={"max_attempts": 0}),
                )
            ) as client:
                return client.converse(**request)

    async def health(self):
        return {
            "connected": self.configured,
            "configured": self.configured,
            "readiness": "verified" if self.verified else "configured",
            "model": self.model_id,
            "transport": "bedrock",
            "message": "GPT6 已配置；首次推理时验证 AWS 访问。"
            if self.configured
            else "GPT6 尚未启用 Bedrock。",
        }

    async def evaluate(self, payload):
        if not self.configured:
            raise DeciderError("GPT6 尚未启用 Bedrock。")
        started = perf_counter()
        request = gpt_request(payload)
        try:
            response = await asyncio.to_thread(self.invoke, request)
            # Bedrock may include opaque binary reasoning blocks. Persist the exact
            # returned choice text and usage, not private/opaque reasoning content.
            message = response["output"]["message"]
            raw = {k: response[k] for k in ("stopReason", "usage", "metrics") if k in response}
            raw["output"] = {
                "message": {
                    "role": message.get("role", "assistant"),
                    "content": [
                        {"text": part["text"]} for part in message["content"] if "text" in part
                    ],
                }
            }
            result = normalize_gpt(raw, payload["questions"])
        except (BotoCoreError, ClientError, OSError) as exc:
            raise DeciderError("GPT6 的 Bedrock 调用失败或超时，本步未执行，可以重试。") from exc
        except (TypeError, KeyError, AttributeError) as exc:
            raise DeciderError("GPT6 响应格式无效，本步未执行。") from exc
        self.verified = True
        return {
            "provider": self.provider,
            "requested_model": self.model_id,
            "payload": payload,
            "provider_request": request,
            "provider_response": raw,
            "provider_response_scope": "choice_text_stop_reason_usage_metrics",
            "response": result,
            "elapsed_ms": round((perf_counter() - started) * 1000, 2),
        }

    async def close(self):
        pass
