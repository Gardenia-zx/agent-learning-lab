# 01-llm-client Notes

## 完成内容

- 封装 `LLMClient`，支持 `base_url`、`api_key`、`model_name`、`timeout`。
- 配置直接从项目根目录 `env.env` 读取（无 `config` 类）。
- 支持返回纯文本响应，或 `return_raw=True` 时返回原始响应与 token 统计。
- 提供 `IntentOutput`（Pydantic）并支持结构化 JSON 校验。
- 提供 `TokenCounter`，记录 `input_tokens`、`output_tokens`、`total_tokens`、`cost_estimate`。
- 新增 `MessageBuilder` 支持链式消息构建。

## 示例

```python
from message import MessageBuilder
from llm_client import LLMClient

messages = (
    MessageBuilder()
    .system("你是一个严谨的助手")
    .user("解释什么是 Agent")
    .build()
)

llm = LLMClient(timeout=20)
text = llm.chat(messages)
print(text)

result = llm.chat(messages, return_raw=True)
print(result.token_usage)
```
