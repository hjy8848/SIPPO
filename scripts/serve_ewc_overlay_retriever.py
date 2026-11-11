#!/usr/bin/env python3
"""Serve one EWC pilot world through the standard JADE retrieval response."""

from __future__ import annotations

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


def normalize(value: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value).casefold()))


class OverlayState:
    def __init__(self, instances: list[dict[str, Any]]) -> None:
        self.instances = {str(row["instance_id"]): row for row in instances}
        self.active_id: str | None = None

    def select(self, instance_id: str) -> None:
        if instance_id not in self.instances:
            raise KeyError(instance_id)
        self.active_id = instance_id

    def retrieve(self, query: str, topk: int) -> list[dict[str, Any]]:
        if self.active_id is None:
            raise RuntimeError("no active EWC instance; POST /select first")
        instance = self.instances[self.active_id]
        documents = instance["world_overlay"].get(
            "retrieval_documents", instance["world_overlay"].get("documents", [])
        )
        query_norm = normalize(query)
        strict_query_matching = bool(instance["world_overlay"].get("strict_query_matching"))
        scored = []
        for document in documents:
            query_forms = [normalize(item) for item in document.get("queries", [])]
            exact = int(query_norm in query_forms)
            if strict_query_matching and not exact:
                continue
            overlap = len(set(query_norm.split()) & set(normalize(document.get("contents", "")).split()))
            scored.append((exact, overlap, document))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [
            {
                "document": {
                    "contents": str(document.get("title", "")) + "\n" + str(document.get("contents", "")),
                    "title": document.get("title", ""),
                    "document_id": document.get("document_id", ""),
                },
                "score": float(exact * 10 + overlap),
            }
            for exact, overlap, document in scored[:topk]
        ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18067)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.instances.read_text(encoding="utf-8").split("\n") if line.strip()]
    state = OverlayState(rows)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *values: object) -> None:
            return

        def send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802
            try:
                size = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(size) or b"{}")
                if self.path == "/select":
                    state.select(str(payload["instance_id"]))
                    self.send_json({"ok": True, "active_instance_id": state.active_id})
                    return
                if self.path == "/retrieve":
                    queries = payload.get("queries", [payload.get("query", "")])
                    result = [state.retrieve(str(query), int(payload.get("topk", 3))) for query in queries]
                    self.send_json({"result": result})
                    return
                self.send_json({"error": "not found"}, 404)
            except Exception as exc:
                self.send_json({"error": str(exc)}, 400)

    print(f"EWC overlay retriever listening on http://{args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
