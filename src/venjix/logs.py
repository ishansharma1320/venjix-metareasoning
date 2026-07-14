"""JSONL episode logging + run manifest.

FOUNDATIONAL SCHEMA (see docs/CLAUDE.md) — record shapes here are load-bearing
for every later baseline, the arbiter, and the plots. Changes must be flagged
loudly, never refactored silently.

Layout: runs/<run_id>/manifest.json + runs/<run_id>/episodes.jsonl, one JSON
object per line, discriminated by "type" ("step" | "shift" | "episode").
`prediction_error` is null until world-model agents exist — reserved now so the
schema never changes. Shift records are runner-side truth, never agent-visible.
"""

import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from venjix.config import RunConfig
from venjix.shifts import ShiftRecord

PACKAGE_VERSION = "0.1.0"


class EpisodeLogger:
    def __init__(self, out_root: str | Path, config: RunConfig):
        config_hash = config.config_hash()
        run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{config_hash[:8]}"
        self.run_dir = Path(out_root) / run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)

        manifest = {
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": config.to_dict(),
            "config_hash": config_hash,
            "package_version": PACKAGE_VERSION,
        }
        (self.run_dir / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2)
        )
        self._file = (self.run_dir / "episodes.jsonl").open("w")

    def _write(self, record: dict) -> None:
        self._file.write(json.dumps(record, sort_keys=True) + "\n")

    def log_step(
        self,
        *,
        episode: int,
        step_in_episode: int,
        global_step: int,
        mode: str,
        action: str,
        parse_error: bool,
        pos: tuple[int, int],
        reward: int,
        done: bool,
        success: bool,
        probe_result,
        llm_calls: int,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        wall_time_ms: float,
        prediction_error: int | None,
        signal_ewma: float | None,
    ) -> None:
        self._write(
            {
                "type": "step",
                "episode": episode,
                "step_in_episode": step_in_episode,
                "global_step": global_step,
                "mode": mode,
                "action": action,
                "parse_error": parse_error,
                "pos": list(pos),
                "reward": reward,
                "done": done,
                "success": success,
                "probe_result": probe_result,
                "llm_calls": llm_calls,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost_usd,
                "wall_time_ms": wall_time_ms,
                # Binary misprediction + the EWMA arbitration signal after this
                # step's update; null for agents that carry no signal.
                "prediction_error": prediction_error,
                "signal_ewma": signal_ewma,
            }
        )

    def log_shift(self, record: ShiftRecord, global_step: int) -> None:
        self._write({"type": "shift", "global_step": global_step, **asdict(record)})

    def log_episode(
        self,
        *,
        episode: int,
        success: bool,
        steps_used: int,
        llm_calls: int,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        wall_time_ms: float,
    ) -> None:
        self._write(
            {
                "type": "episode",
                "episode": episode,
                "success": success,
                "steps_used": steps_used,
                "llm_calls": llm_calls,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost_usd,
                "wall_time_ms": wall_time_ms,
            }
        )

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "EpisodeLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def now_ms() -> float:
    return time.perf_counter() * 1000
