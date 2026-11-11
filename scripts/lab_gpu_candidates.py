#!/usr/bin/env python3
"""List likely-free lab GPUs from the shared gpustat dashboard."""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from dataclasses import asdict, dataclass

try:
    import websocket
except ImportError as exc:
    raise SystemExit(
        "Missing websocket-client. Install it with: pip install websocket-client"
    ) from exc


DEFAULT_URL = "ws://183.129.178.195:16999/ws"
HOST_RE = re.compile(
    r"^(?P<host>\S+)\s+(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+"
)
GPU_RE = re.compile(
    r"^\[(?P<index>\d+)\]\s+"
    r"(?P<name>.*?)\s+\|.*?\|\s*"
    r"(?P<used>\d+)\s*/\s*(?P<total>\d+)\s*MB\s*\|"
    r"(?P<processes>.*)$"
)
TAG_RE = re.compile(r"<[^>]+>")


@dataclass
class GPU:
    host: str
    index: int
    name: str
    used_mib: int
    total_mib: int
    processes: str

    @property
    def free_mib(self) -> int:
        return self.total_mib - self.used_mib

    @property
    def holder_only(self) -> bool:
        process_count = len(
            re.findall(r"\S+:[^/\s]+/\d+\(\d+M\)", self.processes)
        )
        return process_count == 1 and "gpu_holder.py" in self.processes


def plain_text(markup: str) -> str:
    return html.unescape(TAG_RE.sub("", markup)).replace("\xa0", " ").strip()


def fetch_dashboard(url: str, timeout: float) -> str:
    ws = websocket.create_connection(url, timeout=timeout)
    try:
        ws.send("gpustat")
        return ws.recv()
    finally:
        ws.close()


def parse_gpus(markup: str) -> list[GPU]:
    current_host = ""
    current_gpu: GPU | None = None
    gpus: list[GPU] = []

    for raw_line in markup.splitlines():
        line = plain_text(raw_line)
        host_match = HOST_RE.match(line)
        if host_match:
            current_host = host_match.group("host")
            current_gpu = None
            continue

        gpu_match = GPU_RE.match(line)
        if not gpu_match:
            if current_gpu and line:
                current_gpu.processes = (
                    f"{current_gpu.processes} {line}".strip()
                )
            continue
        if not current_host:
            continue

        current_gpu = GPU(
            host=current_host,
            index=int(gpu_match.group("index")),
            name=" ".join(gpu_match.group("name").split()),
            used_mib=int(gpu_match.group("used")),
            total_mib=int(gpu_match.group("total")),
            processes=gpu_match.group("processes").strip(),
        )
        gpus.append(current_gpu)

    return gpus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read the lab GPU dashboard once and list low-memory GPUs without "
            "a reported compute process. Always SSH-verify the selected GPU."
        )
    )
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--max-used-mib", type=int, default=1024)
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Case-insensitive model substring; repeat for multiple models.",
    )
    parser.add_argument(
        "--host",
        action="append",
        default=[],
        help="Host name to include; repeat for multiple hosts.",
    )
    parser.add_argument(
        "--respect-gpu-holder",
        action="store_true",
        help="Exclude GPUs whose only compute process is gpu_holder.py.",
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        markup = fetch_dashboard(args.url, args.timeout)
    except Exception as exc:
        print(f"Failed to read {args.url}: {exc}", file=sys.stderr)
        return 2

    models = [value.casefold() for value in args.model]
    hosts = set(args.host)
    candidates = [
        gpu
        for gpu in parse_gpus(markup)
        if gpu.used_mib <= args.max_used_mib
        and (
            not gpu.processes
            or (gpu.holder_only and not args.respect_gpu_holder)
        )
        and (not models or any(model in gpu.name.casefold() for model in models))
        and (not hosts or gpu.host in hosts)
    ]
    candidates.sort(key=lambda gpu: (-gpu.free_mib, gpu.host, gpu.index))

    if args.json:
        print(
            json.dumps(
                [
                    {**asdict(gpu), "free_mib": gpu.free_mib}
                    for gpu in candidates
                ],
                indent=2,
            )
        )
        return 0

    if not candidates:
        print("No dashboard candidates matched the requested filters.")
        return 1

    print(
        f"{'HOST':<18} {'GPU':>3} {'USED/TOTAL MiB':>19} {'STATUS':>12}  MODEL"
    )
    for gpu in candidates:
        memory = f"{gpu.used_mib}/{gpu.total_mib}"
        status = "holder-only" if gpu.holder_only else "empty"
        print(
            f"{gpu.host:<18} {gpu.index:>3} {memory:>19} "
            f"{status:>12}  {gpu.name}"
        )
    print("\nDashboard candidates only; SSH-verify memory and compute processes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
