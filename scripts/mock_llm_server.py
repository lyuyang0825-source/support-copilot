#!/usr/bin/env python3
"""本地协议桩：模拟一个 OpenAI 兼容的 `/v1/chat/completions`。

它**不是模型**，只是把请求里的“规则答案”原样返回，用来在没有 GPU、没有 API key
的情况下验证 llm 适配层、超时、校验与回退是否接得通。

用法：
    python3 scripts/mock_llm_server.py --port 8123
    export LLM_BASE_URL=http://127.0.0.1:8123/v1
    python3 -m support_copilot.cli -q "退款多久到账" --llm
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 约定
        length = int(self.headers.get("content-length", "0"))
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw)
            user = json.loads(payload["messages"][-1]["content"])
            content = user["规则答案"]
        except Exception:  # noqa: BLE001 - 桩只做协议演示
            content = "收到请求，但无法解析规则答案。"

        body = json.dumps(
            {"choices": [{"message": {"role": "assistant", "content": content}}]},
            ensure_ascii=False,
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8123)
    args = parser.parse_args()
    server = HTTPServer((args.host, args.port), Handler)
    print(f"mock LLM listening on http://{args.host}:{args.port}/v1（不是模型，只回显规则答案）")
    server.serve_forever()


if __name__ == "__main__":
    main()
