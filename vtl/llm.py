"""The single LLM entry point: llm(system, user, model, schema) -> str.

Backends, chosen by name (env VTL_LLM, per stage VTL_LLM_TIMELINE / VTL_LLM_CANDIDATES / VTL_LLM_DECISION):
  ollama:<model>  local Ollama (default ollama:qwen3:4b): JSON-schema structured output, no thinking, temperature 0.
  haiku | sonnet  optional Claude CLI, isolated (no user settings/hooks/MCP/tools/session); only for comparisons.
Every call is appended to STATS (wall time, tokens, $) so stages can report their cost.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import time
import urllib.request

from vtl.common import extract_json

DEFAULT = "ollama:qwen3:4b"
OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
if not OLLAMA_URL.startswith("http"):
    OLLAMA_URL = "http://" + OLLAMA_URL
# Child `claude` must not inherit this session (nesting markers) or the local prompt-rewriting proxy.
CHILD_ENV_DROP = ("ANTHROPIC_BASE_URL", "CLAUDECODE", "AI_AGENT", "_CLAUDE_CODE_ASSUME_FIRST_PARTY_BASE_URL",
                  "CLAUDE_EFFORT", "CLAUDE_PID", "CLAUDE_JOB_DIR")
STATS: list[dict] = []
_LOCK = threading.Lock()


def model_for(stage: str) -> str:
    return os.environ.get(f"VTL_LLM_{stage.upper()}") or os.environ.get("VTL_LLM") or DEFAULT


def is_local(model: str) -> bool:
    return model.startswith("ollama:")


def _record(**rec) -> None:
    with _LOCK:
        STATS.append(rec)


def _ollama(system: str, user: str, name: str, schema: dict | None) -> str:
    # ~3.5 chars per token; leave room for the answer. Small contexts keep the KV cache on the shared GPU.
    need = (len(system) + len(user)) // 3 + 2048
    body = {"model": name, "stream": False, "think": False, "keep_alive": os.environ.get("VTL_OLLAMA_KEEP", "30s"),
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": {"temperature": 0, "seed": 0, "num_ctx": min(32768, -(-need // 2048) * 2048),
                        "num_predict": 2500}}  # a looping answer is cut off (invalid JSON -> one retry)
    if schema:
        body["format"] = schema
    req = urllib.request.Request(f"{OLLAMA_URL}/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        j = json.loads(r.read())
    ev, pe = j.get("eval_count", 0), j.get("prompt_eval_count", 0)
    _record(model="ollama:" + name, wall_s=time.time() - t0, cost_usd=0.0, in_tok=pe, out_tok=ev,
            prefill_tps=pe / max(1e-9, j.get("prompt_eval_duration", 0) / 1e9),
            decode_tps=ev / max(1e-9, j.get("eval_duration", 0) / 1e9))
    return j["message"]["content"]


def _claude(system: str, user: str, name: str, schema: dict | None) -> str:
    if schema:
        system += ("\n\nReply with ONE JSON object that follows this JSON schema, and nothing else:\n"
                   + json.dumps(schema))
    env = {k: v for k, v in os.environ.items()
           if k not in CHILD_ENV_DROP and not k.startswith(("CLAUDECODE", "CLAUDE_CODE_"))}
    cmd = ["claude", "-p", "--model", name, "--output-format", "json", "--safe-mode", "--setting-sources", "project",
           "--strict-mcp-config", "--no-session-persistence", "--tools", "", "--system-prompt", system]
    t0 = time.time()
    with tempfile.TemporaryDirectory() as cwd:  # empty dir: no project CLAUDE.md or settings
        p = subprocess.run(cmd, input=user, capture_output=True, text=True, env=env, cwd=cwd, timeout=900)
    try:
        j = json.loads(p.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"claude CLI failed ({p.returncode}): {(p.stderr or p.stdout)[-400:]}") from None
    u = j.get("usage") or {}
    _record(model=name, wall_s=time.time() - t0, cost_usd=float(j.get("total_cost_usd") or 0.0),
            in_tok=u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0)
            + u.get("cache_creation_input_tokens", 0), out_tok=u.get("output_tokens", 0))
    if j.get("is_error"):
        raise RuntimeError(f"claude CLI error: {str(j.get('result'))[:400]}")
    return j.get("result", "")


def llm(system: str, user: str, model: str, schema: dict | None = None) -> str:
    """One completion. `schema` asks for (Ollama: enforces) a JSON object of that shape."""
    if is_local(model):
        return _ollama(system, user, model.split(":", 1)[1], schema)
    if model in ("haiku", "sonnet", "opus") or model.startswith("claude-"):
        return _claude(system, user, model, schema)
    raise ValueError(f"unknown LLM backend {model!r} (use ollama:<model>, haiku or sonnet)")


def llm_json(system: str, user: str, model: str, schema: dict | None = None, check=None, call=None) -> dict:
    """llm() + JSON extraction (+ optional check(dict) raising ValueError), with one retry on failure."""
    call = call or llm
    err = None
    for attempt in range(2):
        prompt = user if attempt == 0 else f"{user}\n\n(Your previous reply was invalid: {err}. Reply with the JSON object only.)"
        try:
            data = extract_json(call(system, prompt, model, schema))
            if check:
                check(data)
            return data
        except (ValueError, RuntimeError, OSError) as e:  # bad JSON, CLI or HTTP failure
            err = str(e)[:200]
    raise RuntimeError(f"LLM ({model}) gave no valid JSON after a retry: {err}")


def unload(model: str) -> None:
    """Free the GPU right away (Ollama otherwise holds the model for keep_alive)."""
    if not is_local(model):
        return
    body = {"model": model.split(":", 1)[1], "keep_alive": 0, "messages": []}
    try:
        urllib.request.urlopen(urllib.request.Request(f"{OLLAMA_URL}/api/chat", data=json.dumps(body).encode(),
                                                      headers={"Content-Type": "application/json"}), timeout=30)
    except OSError:
        pass


def summary(since: int = 0) -> dict:
    rows = STATS[since:]
    return {"calls": len(rows), "wall_s": round(sum(r["wall_s"] for r in rows), 1),
            "cost_usd": round(sum(r["cost_usd"] for r in rows), 4),
            "in_tok": sum(r["in_tok"] for r in rows), "out_tok": sum(r["out_tok"] for r in rows)}
