import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Type, TypeVar

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError
from openai import APITimeoutError, APIConnectionError, RateLimitError, APIStatusError

try:
    from .token_counter import TokenCounter, TokenUsage
except ImportError:
    from token_counter import TokenCounter, TokenUsage


# 泛型T规定T的类型只能是BaseModel或其子类，确保结构化输出解析的类型安全。
T = TypeVar("T", bound=BaseModel)


class LLMClientError(Exception):
    """LLM 客户端通用异常基类。

    所有与请求执行、响应处理相关的业务异常都建议继承该类型，
    便于调用方统一捕获与日志分类。
    """

    pass


class LLMTimeoutError(LLMClientError):
    """LLM 请求超时异常。"""

    pass


class StructuredOutputError(LLMClientError):
    """结构化输出解析或校验失败异常。"""

    pass


class IntentOutput(BaseModel):
    """意图识别结构化输出模型。

    字段说明：
    - intent: 模型判定的意图标签，如 search、qa 等
    - confidence: 置信度，通常在 0~1 区间
    - reason: 模型给出的简要原因说明
    """

    intent: str
    confidence: float
    reason: str


@dataclass
class LLMCallResult:
    """LLM 调用结果容器。

    text: 纯文本回复
    raw_response: SDK 原始响应对象
    token_usage: token 统计结果
    """

    text: str
    raw_response: Any
    token_usage: TokenUsage


def load_project_env() -> None:
    """从项目根目录加载 env.env 到进程环境变量。

    该函数允许调用方在不手动 export 环境变量的情况下，
    直接通过 env.env 配置 api_key、base_url、model 等参数。
    """

    env_path = Path(__file__).resolve().parents[2] / "env.env"
    load_dotenv(env_path)


def _extract_json_block(text: str) -> str:
    """从模型输出文本中提取首个 JSON 对象字符串。

    兼容两种常见输出：
    1. 仅包含 JSON 的纯文本
    2. 前后夹杂自然语言说明的文本
    """

    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise StructuredOutputError("No JSON object found in model output")
    return text[start : end + 1]


class LLMClient:
    """统一封装 LLM 对话调用。

    支持能力：
    - 从参数或 env.env 读取 api_key/base_url/model_name
    - 自定义超时 timeout
    - 返回纯文本或原始响应
    - 异常统一封装
    - 结构化输出解析（Pydantic 校验）
    - token 与费用估算记录
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: float = 30.0,
        client: Optional[Any] = None,
    ) -> None:
        """初始化客户端实例。

        参数说明：
        - api_key: API 密钥，优先级高于环境变量
        - base_url: 兼容 OpenAI 协议服务的网关地址
        - model_name: 调用模型名称
        - timeout: 请求超时时间（秒）
        - client: 可注入的 SDK 客户端，便于测试 mock
        """

        load_project_env()
        self.api_key = api_key or os.getenv("api_key") or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("base_url") or os.getenv("OPENAI_BASE_URL")
        self.model_name = model_name or os.getenv("model") or os.getenv("OPENAI_MODEL")
        self.timeout = timeout
        self.token_counter = TokenCounter()

        if not self.api_key:
            raise ValueError("api_key is required")
        if not self.model_name:
            raise ValueError("model_name is required")

        self.client = client or OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    def chat(self, messages: list[dict[str, str]], return_raw: bool = False, **kwargs: Any) -> Any:
        """发送聊天请求并返回文本或完整结果。

        参数说明：
        - messages: OpenAI chat 格式消息列表
        - return_raw: 为 True 时返回 LLMCallResult，否则仅返回文本
        - kwargs: 透传到 chat.completions.create 的额外参数，就是chat函数可以包含其他参数，如 temperature、max_tokens 等
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **kwargs,
            )
            # 选取模型返回的第一条文本回复进行处理，兼容多条回复的情况。
            text = response.choices[0].message.content or ""
            usage_obj = getattr(response, "usage", None)
            token_usage = self.token_counter.record(
                input_text="\n".join(m.get("content", "") for m in messages),
                output_text=text,
                usage_obj=usage_obj,
                model_name=self.model_name,
            )
            result = LLMCallResult(text=text, raw_response=response, token_usage=token_usage)
            return result if return_raw else result.text
        except APITimeoutError as exc:
            raise LLMTimeoutError(f"LLM request timeout: {exc}") from exc
        except RateLimitError as exc:
            raise LLMClientError(f"LLM rate limit exceeded: {exc}") from exc
        except APIConnectionError as exc:
            raise LLMClientError(f"LLM connection failed: {exc}") from exc
        except APIStatusError as exc:
            raise LLMClientError(f"LLM API status error: {exc}") from exc
        except Exception as exc:
            raise LLMClientError(f"Unexpected LLM request failed: {exc}") from exc

    def chat_structured(self, messages: list[dict[str, str]], schema: Type[T], **kwargs: Any) -> T:
        """发送请求并将返回内容解析为指定 Pydantic 模型。"""

        text = self.chat(messages=messages, return_raw=False, **kwargs)
        return self.parse_structured_text(text, schema)

    @staticmethod
    def parse_structured_text(text: str, schema: Type[T]) -> T:
        """解析文本中的 JSON 并用 Pydantic 进行结构校验。"""

        try:
            payload = _extract_json_block(text)
            return schema.model_validate_json(payload)
        except ValidationError as exc:
            raise StructuredOutputError(f"Structured output validation failed: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise StructuredOutputError(f"JSON decode failed: {exc}") from exc


if __name__ == "__main__":
    try:
        from .message import MessageBuilder
    except ImportError:
        from message import MessageBuilder

    messages = (
        MessageBuilder()
        .system("你是一个严谨的助手")
        .user("解释什么是 Agent")
        .build()
    )

    llm = LLMClient(timeout=20)
    reply = llm.chat(messages)
    print(reply)
