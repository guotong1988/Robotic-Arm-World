"""用微调后的 Qwen 做 Lift 测评，直连 vLLM 的 OpenAI chat completions。

    python3 eval_lift_mllm_vllm.py --api-base http://127.0.0.1:8000/v1 --episodes 20

vLLM 服务示例：

    python3 -m vllm.entrypoints.openai.api_server \\
        --model /the_model_path/ --host 0.0.0.0 --port 8000

    python3 eval_lift_mllm_vllm.py --oracle --episodes 5
"""

import argparse
import json
import urllib.error
import urllib.request

from eval_lift_core import (
    OracleClient,
    ServerError,
    add_common_args,
    chat_messages,
    post_chat,
    run_eval,
    self_test,
)


class VLLMClient:
    def __init__(self, api_base, model, timeout, max_tokens, temperature, system, text_template):
        self.api_base = api_base.rstrip("/")
        if not self.api_base.endswith("/v1"):
            self.api_base += "/v1"
        self.model = model or ""
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system = system
        self.text_template = text_template
        self._drop_thinking_flag = False

    def _request(self, method, path, payload=None):
        url = self.api_base + path
        data = None
        headers = {"Content-Type": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ServerError("HTTP {} {}\n{}".format(exc.code, url, detail[:800])) from exc
        except urllib.error.URLError as exc:
            raise ServerError("连不上 {} ：{}".format(url, exc.reason)) from exc

    def resolve_model(self):
        if self.model:
            return self.model
        data = self._request("GET", "/models")
        models = data.get("data") or []
        if not models:
            raise ServerError("{} 没有返回可用模型".format(self.api_base))
        self.model = models[0]["id"]
        return self.model

    def complete(self, image, instruction):
        messages = chat_messages(image, instruction, self.system, self.text_template)
        drop_thinking = [self._drop_thinking_flag]

        def request(payload):
            return self._request("POST", "/chat/completions", payload)

        text = post_chat(
            request,
            self.model,
            messages,
            self.temperature,
            self.max_tokens,
            drop_thinking,
        )
        self._drop_thinking_flag = drop_thinking[0]
        return text


def main():
    parser = argparse.ArgumentParser(description="直连 vLLM，按抓取计划在 robosuite Lift 上测成功率")
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1", help="vLLM OpenAI 接口地址")
    add_common_args(parser)
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    if args.oracle:
        client = OracleClient()
    else:
        client = VLLMClient(
            args.api_base,
            args.model,
            args.timeout,
            args.max_tokens,
            args.temperature,
            args.system,
            args.text_template,
        )
        model = client.resolve_model()
        print("api {} model {}".format(client.api_base, model), flush=True)

    run_eval(
        args,
        client,
        "vllm",
        extra_summary=lambda item: {"api_base": item.api_base},
    )


if __name__ == "__main__":
    main()
