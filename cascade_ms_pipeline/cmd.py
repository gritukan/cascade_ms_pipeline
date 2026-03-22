from __future__ import annotations

import datetime as dt
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence


def _quote(text: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_./:=+\-]+", text):
        return text
    return "'" + text.replace("'", "'\"'\"'") + "'"


def run_cmd(
    cmd: Sequence[str],
    *,
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    dry_run: bool = False,
    log_path: Optional[Path] = None,
) -> None:
    cmd = [str(x) for x in cmd]
    cmd_str = " ".join(_quote(x) for x in cmd)
    msg = f"[{dt.datetime.now().astimezone().isoformat(timespec='seconds')}] $ {cmd_str}"
    print(msg)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(msg + "\n")
    if dry_run:
        return

    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines: List[str] = []
    assert proc.stdout is not None
    with proc.stdout:
        for line in iter(proc.stdout.readline, ""):
            print(line, end="")
            sys.stdout.flush()
            lines.append(line)
            if log_path is not None:
                with log_path.open("a", encoding="utf-8") as fh:
                    fh.write(line)
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(
            f"Command failed with exit code {rc}\nCommand: {cmd_str}\nOutput:\n{''.join(lines)}"
        )


def capture_cmd(
    cmd: Sequence[str],
    *,
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    timeout: Optional[int] = 30,
) -> str:
    proc = subprocess.run(
        [str(x) for x in cmd],
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
        check=False,
    )
    return proc.stdout.strip()
