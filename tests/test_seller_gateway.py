import json

import httpx
import pytest

from app import seller_gateway as gw


def upstream_sse(chunks: list[str], finish: str = "length") -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        handler.seen = json.loads(request.content)
        lines = [json.dumps({"choices": [{"delta": {"content": c}, "finish_reason": None}]}) for c in chunks]
        lines.append(json.dumps({"choices": [{"delta": {}, "finish_reason": finish}]}))
        body = "".join(f"data: {line}\n\n" for line in lines) + "data: [DONE]\n\n"
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    return httpx.MockTransport(handler), handler


@pytest.fixture(autouse=True)
def config(monkeypatch):
    monkeypatch.setattr(gw.cfg, "upstream", "http://vllm.local/v1")
    monkeypatch.setattr(gw.cfg, "upstream_model", "meta-llama/Llama-3.1-70B-Instruct")
    monkeypatch.setattr(gw.cfg, "chat_continuation", "vllm")


async def collect(transport, body):
    async with httpx.AsyncClient(transport=transport) as http:
        return [json.loads(chunk[6:]) async for chunk in gw.relay(http, body)]


async def test_relay_indexes_tokens_from_resume_point_and_continues_prefix():
    transport, handler = upstream_sse([" world", "!", " more"])
    body = {
        "job_id": "j",
        "instrument": "x",
        "prompt": "p",
        "max_tokens": 12,
        "resume_from": 10,
        "prefix": "Hello",
        "messages": [{"role": "user", "content": "hi"}],
    }
    events = await collect(transport, body)
    # The segment bound (12) caps output even though upstream sent three chunks.
    assert events == [{"index": 10, "token": " world"}, {"index": 11, "token": "!"}, {"done": True, "finish_reason": "length"}]
    assert handler.seen["max_tokens"] == 2
    assert handler.seen["messages"][-1] == {"role": "assistant", "content": "Hello"}
    assert handler.seen["continue_final_message"] is True and handler.seen["add_generation_prompt"] is False


async def test_relay_natural_stop_and_completions_api():
    transport, handler = upstream_sse(["a", "b"], finish="stop")
    events = await collect(transport, {"job_id": "j", "instrument": "x", "prompt": "p", "max_tokens": 50, "resume_from": 0})
    assert events[-1] == {"done": True, "finish_reason": "stop"} and len(events) == 3
    assert handler.seen["prompt"] == "p" and handler.seen["max_tokens"] == 50


def test_refuses_to_resell_proprietary_apis(monkeypatch):
    for name, value in [
        ("api_key", "k"),
        ("public_url", "https://x"),
        ("instrument", "i"),
        ("upstream", "https://api.openai.com/v1"),
    ]:
        monkeypatch.setattr(gw.cfg, name, value)
    monkeypatch.setattr(gw.cfg, "allow_proprietary", False)
    with pytest.raises(SystemExit, match="refusing to resell"):
        gw.check_config(gw.cfg)
    monkeypatch.setattr(gw.cfg, "upstream", "http://127.0.0.1:8001/v1")
    gw.check_config(gw.cfg)
