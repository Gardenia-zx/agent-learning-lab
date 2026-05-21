from dataclasses import dataclass
from typing import Any


@dataclass
class TokenUsage:
    """token 使用统计结果。

    字段说明：
    - input_tokens: 输入 token 数
    - output_tokens: 输出 token 数
    - total_tokens: 总 token 数
    - cost_estimate: 基于费率估算的费用
    """

    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_estimate: float


class TokenCounter:
    """token 统计与费用估算工具。

    优先使用模型响应中的 usage 字段；若缺失 usage，
    则采用字符长度做近似估算，满足开发阶段的快速统计需求。
    """

    # 简单估算：按 1 token ~= 4 chars。
    CHARS_PER_TOKEN = 4

    # 这里用示意费率，实际可按模型官方费率替换
    MODEL_RATES = {
        "default": {"input": 0.001, "output": 0.002},
    }

    @classmethod
    def estimate_tokens(cls, text: str) -> int:
        """基于字符数估算 token 数。"""

        if not text:
            return 0
        return max(1, len(text) // cls.CHARS_PER_TOKEN)

    @classmethod
    def _get_rates(cls, model_name: str | None) -> dict[str, float]:
        """获取模型对应费率，不存在时回退 default。"""

        if not model_name:
            return cls.MODEL_RATES["default"]
        return cls.MODEL_RATES.get(model_name, cls.MODEL_RATES["default"])

    def record(
        self,
        input_text: str,
        output_text: str,
        usage_obj: Any = None,
        model_name: str | None = None,
    ) -> TokenUsage:
        """记录一次调用的 token 与费用。

        参数说明：
        - input_text/output_text: 用于无 usage 时的估算输入
        - usage_obj: SDK 返回的 usage 对象（可选）
        - model_name: 用于匹配费率
        """

        if usage_obj is not None:
            # 模型返回了usage就读取，没有属性默认为第三个参数
            input_tokens = int(getattr(usage_obj, "prompt_tokens", 0))
            output_tokens = int(getattr(usage_obj, "completion_tokens", 0))
            total_tokens = int(
                getattr(usage_obj, "total_tokens", input_tokens + output_tokens)
            )
        else:
            input_tokens = self.estimate_tokens(input_text)
            output_tokens = self.estimate_tokens(output_text)
            total_tokens = input_tokens + output_tokens

        rates = self._get_rates(model_name)
        cost_estimate = (
            input_tokens / 1000 * rates["input"] + output_tokens / 1000 * rates["output"]
        )

        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            # 保留8位小数，避免浮点数精度问题
            cost_estimate=round(cost_estimate, 8),
        )
