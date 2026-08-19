import ast
import json
import re
import time
import logging

from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Union,
)
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


'''
自定义异常

将连接错误和生成错误分开，有利于后续针对不同问题进行处理。
'''

class OllamaConnectionError(RuntimeError):
    """Ollama 服务连接异常。
       用于表示：
         Ollama 没有启动；
         地址配置错误；
         请求超时；
         模型不存在；
         Ollama 返回 HTTP 错误。
    """


class OllamaGenerationError(RuntimeError):
    """Ollama 文本生成异常。
       用于表示：
         模型生成失败；
         Ollama 返回错误；
         模型返回空内容；
         草稿生成失败。
    """


logger = logging.getLogger(
    "medical_generation"
)


##LLMGenerator本地集成

class LLMGenerator:
    """
    基于 Ollama /api/chat 接口的本地大模型生成器。

    Args:
        model_name:
            Ollama 模型名称，例如 qwen2.5:7b。

        base_url:
            Ollama 服务地址。

        timeout:
            单次 HTTP 请求超时时间，单位为秒。

        default_temperature:
            未显式指定时使用的默认温度。

        default_max_tokens:
            未显式指定时使用的最大输出 token 数。

        keep_alive:
            模型在 Ollama 内存中的保留时间。

        test_connection:
            初始化时是否检测 Ollama 服务和模型。
    """

    def __init__(
        self,
        model_name: str = "deepseek-r1-7b",
        base_url: str = "http://localhost:11434",
        timeout: int = 300,
        default_temperature: float = 0.2,
        default_max_tokens: int = 1600,
        keep_alive: Union[str, int] = "10m",
        test_connection: bool = True,
    ):
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name不能为空。")

        if timeout <= 0:
            raise ValueError("timeout必须大于0。")

        if not 0.0 <= default_temperature <= 2.0:
            raise ValueError(
                "default_temperature必须在[0, 2]之间。"
            )

        if default_max_tokens <= 0:
            raise ValueError(
                "default_max_tokens必须大于0。"
            )

        self.model_name = model_name.strip()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        self.default_temperature = (
            default_temperature
        )

        self.default_max_tokens = (
            default_max_tokens
        )

        self.keep_alive = keep_alive

        self.last_health_check = None

        if test_connection:
            self.last_health_check = (
                self.test_connection()
            )

    # --------------------------------------------------------
    # HTTP 请求
    # --------------------------------------------------------

    def _request_json(
        self,
        endpoint: str,
        payload: Optional[Dict[str, Any]] = None,
        method: str = "POST",
    ) -> Dict[str, Any]:
        """
        向 Ollama 发送 JSON 请求。
        """

        url = (
            f"{self.base_url}/"
            f"{endpoint.lstrip('/')}"
        )

        data = None

        if payload is not None:
            data = json.dumps(
                payload,
                ensure_ascii=False,
            ).encode("utf-8")

        request = Request(
            url=url,
            data=data,
            method=method,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )

        try:
            with urlopen(
                request,
                timeout=self.timeout,
            ) as response:
                response_text = (
                    response.read()
                    .decode("utf-8")
                )

        except HTTPError as exc:
            try:
                error_body = (
                    exc.read()
                    .decode("utf-8")
                )
            except Exception:
                error_body = ""

            raise OllamaConnectionError(
                f"Ollama HTTP错误："
                f"{exc.code} {exc.reason}。"
                f"{error_body}"
            ) from exc

        except URLError as exc:
            raise OllamaConnectionError(
                f"无法连接Ollama服务："
                f"{self.base_url}。"
                "请确认Ollama已经启动。"
                f"原始错误：{exc.reason}"
            ) from exc

        except TimeoutError as exc:
            raise OllamaConnectionError(
                f"Ollama请求超时："
                f"{self.timeout}秒。"
            ) from exc

        try:
            return json.loads(response_text)

        except json.JSONDecodeError as exc:
            raise OllamaConnectionError(
                "Ollama返回了非JSON响应："
                f"{response_text[:500]}"
            ) from exc

    # --------------------------------------------------------
    # 初始化连接测试
    # --------------------------------------------------------

    def test_connection(
        self,
        require_model: bool = True,
    ) -> Dict[str, Any]:
        """
        检测 Ollama 服务是否可用，并检查模型是否存在。
        """

        start_time = time.perf_counter()

        response = self._request_json(
            endpoint="/api/tags",
            method="GET",
        )

        models = response.get(
            "models",
            [],
        )

        model_names = {
            str(
                item.get(
                    "name",
                    item.get("model", ""),
                )
            ).strip()
            for item in models
        }

        model_names.discard("")

        model_available = (
            self.model_name in model_names
            or (
                ":" not in self.model_name
                and f"{self.model_name}:latest"
                in model_names
            )
        )

        if (
            require_model
            and not model_available
        ):
            available_text = (
                ", ".join(sorted(model_names))
                if model_names
                else "无"
            )

            raise OllamaConnectionError(
                f"Ollama服务正常，但未找到模型"
                f"“{self.model_name}”。"
                f"当前模型：{available_text}"
            )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        result = {
            "connected": True,
            "model_available": model_available,
            "model_name": self.model_name,
            "available_models": sorted(
                model_names
            ),
            "latency_seconds": round(
                elapsed,
                4,
            ),
        }

        logger.info(
            "Ollama连接成功 | model=%s | latency=%.4fs",
            self.model_name,
            elapsed,
        )

        return result

    # --------------------------------------------------------
    # JSON 提取与修复
    # --------------------------------------------------------

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        """
        删除 Markdown JSON 代码块标记。
        """

        text = str(text or "").strip()

        fenced_match = re.search(
            r"```(?:json)?\s*([\s\S]*?)```",
            text,
            flags=re.IGNORECASE,
        )

        if fenced_match:
            return (
                fenced_match
                .group(1)
                .strip()
            )

        return text

    @staticmethod
    def _raw_decode_json(
        text: str,
    ) -> Optional[Any]:
        """
        从混合文本中查找第一个可解析的 JSON 对象或数组。
        """

        decoder = json.JSONDecoder()

        start_positions = [
            match.start()
            for match in re.finditer(
                r"[\{\[]",
                text,
            )
        ]

        for start in start_positions:
            try:
                value, _ = decoder.raw_decode(
                    text[start:]
                )

                return value

            except json.JSONDecodeError:
                continue

        return None

    @staticmethod
    def _balance_json_symbols(
        text: str,
    ) -> str:
        """
        补充缺失的引号、花括号和方括号。

        只对字符串之外的括号进行统计。
        """

        output = []
        stack = []

        in_string = False
        escaped = False

        matching = {
            "}": "{",
            "]": "[",
        }

        closing = {
            "{": "}",
            "[": "]",
        }

        for char in text:
            output.append(char)

            if escaped:
                escaped = False
                continue

            if char == "\\" and in_string:
                escaped = True
                continue

            if char == '"':
                in_string = not in_string
                continue

            if in_string:
                continue

            if char in "{[":
                stack.append(char)

            elif char in "}]":
                if (
                    stack
                    and stack[-1]
                    == matching[char]
                ):
                    stack.pop()

        if in_string:
            output.append('"')

        while stack:
            output.append(
                closing[stack.pop()]
            )

        return "".join(output)

    @classmethod
    def _repair_json_text(
        cls,
        text: str,
    ) -> str:
        """
        对常见 JSON 格式错误进行轻量修复。
        """

        candidate = cls._strip_code_fence(
            text
        )

        # 中文全角符号替换
        replacements = {
            "“": '"',
            "”": '"',
            "：": ":",
            "，": ",",
        }

        for old, new in replacements.items():
            candidate = candidate.replace(
                old,
                new,
            )

        # 删除 JSON 前面的解释文本
        positions = [
            position
            for position in (
                candidate.find("{"),
                candidate.find("["),
            )
            if position >= 0
        ]

        if positions:
            candidate = candidate[
                min(positions):
            ]

        # 给未加引号的简单键名补充引号
        candidate = re.sub(
            r"([{,]\s*)"
            r"([A-Za-z_][A-Za-z0-9_\-]*)"
            r"\s*:",
            r'\1"\2":',
            candidate,
        )

        # 删除闭合符号前的多余逗号
        candidate = re.sub(
            r",\s*([}\]])",
            r"\1",
            candidate,
        )

        return cls._balance_json_symbols(
            candidate
        )

    @classmethod
    def extract_json(
        cls,
        text: str,
    ) -> Optional[Any]:
        """
        从模型输出中提取 JSON。

        解析顺序：
        1. 直接解析；
        2. 从混合文本中扫描JSON；
        3. 修复常见格式问题；
        4. 使用literal_eval兼容Python字典形式。
        """

        cleaned = cls._strip_code_fence(
            text
        )

        try:
            return json.loads(cleaned)

        except json.JSONDecodeError:
            pass

        decoded = cls._raw_decode_json(
            cleaned
        )

        if decoded is not None:
            return decoded

        repaired = cls._repair_json_text(
            cleaned
        )

        try:
            return json.loads(repaired)

        except json.JSONDecodeError:
            pass

        try:
            literal_value = ast.literal_eval(
                repaired
            )

            if isinstance(
                literal_value,
                (dict, list),
            ):
                return literal_value

        except (
            SyntaxError,
            ValueError,
        ):
            pass

        return None

    # --------------------------------------------------------
    # 单次生成
    # --------------------------------------------------------

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        require_json: bool = False,
        json_schema: Optional[Dict[str, Any]] = None,
        extra_options: Optional[
            Dict[str, Any]
        ] = None,
        think: bool = False,
    ) -> Dict[str, Any]:
        """
        调用 Ollama 生成文本。

        Args:
            prompt:
                用户提示词。

            system_prompt:
                系统提示词。

            temperature:
                当前请求温度。

            max_tokens:
                当前请求最大输出token数。

            require_json:
                是否要求JSON输出。

            json_schema:
                可选JSON Schema。
                传入时优先使用Schema约束。

            extra_options:
                其他Ollama options参数。
        """

        if not isinstance(prompt, str):
            raise TypeError(
                "prompt必须是字符串。"
            )

        prompt = prompt.strip()

        if not prompt:
            raise ValueError(
                "prompt不能为空。"
            )

        temperature = (
            self.default_temperature
            if temperature is None
            else temperature
        )

        max_tokens = (
            self.default_max_tokens
            if max_tokens is None
            else max_tokens
        )

        if not 0.0 <= temperature <= 2.0:
            raise ValueError(
                "temperature必须在[0, 2]之间。"
            )

        if max_tokens <= 0:
            raise ValueError(
                "max_tokens必须大于0。"
            )

        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": (
                    str(system_prompt).strip()
                ),
            })

        if require_json:
            json_instruction = (
                "\n\n请只返回合法JSON，不要添加"
                "Markdown代码块、解释文字或JSON之外"
                "的其他内容。"
            )

            if json_schema:
                json_instruction += (
                    "\n必须符合以下JSON Schema：\n"
                    + json.dumps(
                        json_schema,
                        ensure_ascii=False,
                        indent=2,
                    )
                )

            prompt += json_instruction

        messages.append({
            "role": "user",
            "content": prompt,
        })

        options = {
            "temperature": temperature,
            "num_predict": int(max_tokens),
        }

        if extra_options:
            options.update(
                extra_options
            )

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "keep_alive": self.keep_alive,
            "think": think,
            "options": options,
        }

        if json_schema is not None:
            payload["format"] = json_schema

        elif require_json:
            payload["format"] = "json"

        start_time = time.perf_counter()

        logger.info(
            "开始生成 | model=%s | temperature=%.2f "
            "| max_tokens=%d | json=%s",
            self.model_name,
            temperature,
            max_tokens,
            require_json,
        )

        response = self._request_json(
            endpoint="/api/chat",
            payload=payload,
            method="POST",
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        if "error" in response:
            raise OllamaGenerationError(
                str(response["error"])
            )
        print(
            json.dumps(
                response,
                ensure_ascii=False,
                indent=2,
            )
        )
        message = response.get(
            "message",
            {},
        )

        text = str(
            message.get(
                "content",
                "",
            )
        ).strip()

        if not text:
            raise OllamaGenerationError(
                "Ollama返回了空文本。"
            )

        parsed_json = None

        if require_json:
            parsed_json = self.extract_json(
                text
            )

        metrics = {
            "elapsed_seconds": round(
                elapsed,
                4,
            ),
            "prompt_tokens": int(
                response.get(
                    "prompt_eval_count",
                    0,
                )
                or 0
            ),
            "output_tokens": int(
                response.get(
                    "eval_count",
                    0,
                )
                or 0
            ),
            "total_duration_seconds": (
                float(
                    response.get(
                        "total_duration",
                        0,
                    )
                    or 0
                )
                / 1_000_000_000
            ),
            "load_duration_seconds": (
                float(
                    response.get(
                        "load_duration",
                        0,
                    )
                    or 0
                )
                / 1_000_000_000
            ),
            "done_reason": response.get(
                "done_reason"
            ),
        }

        logger.info(
            "生成完成 | elapsed=%.3fs "
            "| prompt_tokens=%d "
            "| output_tokens=%d "
            "| chars=%d",
            elapsed,
            metrics["prompt_tokens"],
            metrics["output_tokens"],
            len(text),
        )

        return {
            "text": text,
            "parsed_json": parsed_json,
            "metrics": metrics,
            "raw_response": response,
        }

    # --------------------------------------------------------
    # 批量生成
    # --------------------------------------------------------

    def batch_generate(
        self,
        prompts: Sequence[
            Union[str, Mapping[str, Any]]
        ],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        require_json: bool = False,
        json_schema: Optional[
            Dict[str, Any]
        ] = None,
        max_workers: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        批量生成。

        prompts中的元素可以是：
        1. 字符串；
        2. generate()参数字典。
        """

        if max_workers <= 0:
            raise ValueError(
                "max_workers必须大于0。"
            )

        def build_kwargs(item):
            base_kwargs = {
                "system_prompt": system_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "require_json": require_json,
                "json_schema": json_schema,
            }

            if isinstance(item, str):
                base_kwargs["prompt"] = item
                return base_kwargs

            if isinstance(item, Mapping):
                merged = dict(base_kwargs)
                merged.update(dict(item))
                return merged

            raise TypeError(
                "批量输入必须是字符串或字典。"
            )

        request_items = [
            build_kwargs(item)
            for item in prompts
        ]

        # 顺序模式便于显存较小的设备使用
        if max_workers == 1:
            return [
                self.generate(**kwargs)
                for kwargs in request_items
            ]

        results = [None] * len(
            request_items
        )

        with ThreadPoolExecutor(
            max_workers=max_workers
        ) as executor:
            future_map = {
                executor.submit(
                    self.generate,
                    **kwargs,
                ): index
                for index, kwargs
                in enumerate(request_items)
            }

            for future in as_completed(
                future_map
            ):
                index = future_map[future]
                results[index] = future.result()

        return results