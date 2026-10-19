"""vtl index --exclusive-gpu: pause other GPU users for the run and give the local timeline LLM the whole GPU.

The user's other GPU service (systemd user unit semsearch.service) is stopped for the duration and started again in
a `finally`, so it comes back after errors and Ctrl-C too (SIGTERM is turned into SystemExit for the same reason).
Only services that were running are stopped, and only those are restarted."""
from __future__ import annotations

import signal
import subprocess
import threading
from contextlib import contextmanager

from vtl import llm as L

SERVICES = ("semsearch.service",)


def _systemctl(run, *args: str):
    return run(["systemctl", "--user", *args], capture_output=True, text=True)


@contextmanager
def exclusive(services=SERVICES, log=print, run=subprocess.run):
    """Stop the running `services` (systemctl --user) for the body; restart them afterwards whatever happens."""
    stopped: list[str] = []
    main = threading.current_thread() is threading.main_thread()
    old = signal.getsignal(signal.SIGTERM) if main else None
    if main:
        def on_term(signum, frame):
            raise SystemExit(128 + signum)

        signal.signal(signal.SIGTERM, on_term)
    try:
        for s in services:
            if _systemctl(run, "is-active", "--quiet", s).returncode != 0:
                log(f"exclusive GPU: {s} is not running, left alone")
                continue
            r = _systemctl(run, "stop", s)
            stopped.append(s)  # restart even if stop reported an error: it may be half-stopped
            log(f"exclusive GPU: stopped {s}" if r.returncode == 0 else
                f"exclusive GPU: stopping {s} failed ({r.returncode}): {(r.stderr or '').strip()[:200]}")
        yield stopped
    finally:
        for s in reversed(stopped):
            r = _systemctl(run, "start", s)
            log(f"exclusive GPU: restarted {s}" if r.returncode == 0 else
                f"exclusive GPU: FAILED to restart {s} ({r.returncode}): {(r.stderr or '').strip()[:200]} "
                f"-- run: systemctl --user start {s}")
        if main:
            signal.signal(signal.SIGTERM, old)


def free_mib() -> int | None:
    """Free GPU memory as seen by this process (None without CUDA)."""
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.mem_get_info()[0] // 2**20)
    except Exception:  # noqa: BLE001 - only a log line depends on it
        return None
    return None


def ollama_alone(model: str, log=print) -> None:
    """Before the timeline stage: unload every other Ollama model and offload all of `model`'s layers to the GPU."""
    if not L.is_local(model):
        return
    name = model.split(":", 1)[1]
    for other in L.running():
        if other != name:
            L.unload("ollama:" + other)
            log(f"exclusive GPU: unloaded Ollama model {other}")
    L.OLLAMA_OPTIONS["num_gpu"] = 999
    log(f"exclusive GPU: {name} with all layers on the GPU (free now: {free_mib()} MiB)")
