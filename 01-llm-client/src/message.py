from dataclasses import dataclass
from typing import ClassVar, Iterable, List

@dataclass(frozen=True)
class Message:
	"""单条对话消息对象。

	设计目标：
	- 统一 role/content 数据结构
	- 在创建阶段完成合法性校验
	- 提供与 OpenAI Chat API 兼容的字典转换方法
	"""

	role: str
	content: str
	# 允许的角色集合，用于输入校验。
	ALLOWED_ROLES: ClassVar[set[str]] = {"system", "user", "assistant", "tool"}

	def __post_init__(self) -> None:
		"""实例创建后执行字段校验。"""

		if self.role not in self.ALLOWED_ROLES:
			raise ValueError(f"Invalid role: {self.role}")
		# content 需为非空字符串，避免把空消息传给模型。
		if not isinstance(self.content, str) or not self.content.strip():
			raise ValueError("content must be a non-empty string")

	def to_dict(self) -> dict[str, str]:
		"""将单条消息转换为 API 所需字典格式。"""

		return {"role": self.role, "content": self.content}

	@classmethod
	def to_dict_list(cls, messages: Iterable["Message"]) -> List[dict[str, str]]:
		"""将一组 Message 批量转换为字典列表。"""

		return [msg.to_dict() for msg in messages]


class MessageBuilder:
	"""链式构造消息列表的辅助类。

	示例：
	messages = (
		MessageBuilder()
		.system("你是一个严谨的助手")
		.user("解释什么是 Agent")
		.build()
	)
	"""

	def __init__(self) -> None:
		"""初始化内部消息缓存。"""

		self._messages: List[Message] = []

	def _add(self, role: str, content: str) -> "MessageBuilder":
		"""追加一条消息并返回自身，支持链式调用。"""

		self._messages.append(Message(role=role, content=content))
		return self

	def system(self, content: str) -> "MessageBuilder":
		"""追加 system 消息。"""

		return self._add("system", content)

	def user(self, content: str) -> "MessageBuilder":
		"""追加 user 消息。"""

		return self._add("user", content)

	def assistant(self, content: str) -> "MessageBuilder":
		"""追加 assistant 消息。"""

		return self._add("assistant", content)

	def tool(self, content: str) -> "MessageBuilder":
		"""追加 tool 消息。"""

		return self._add("tool", content)

	def build(self) -> List[dict[str, str]]:
		"""导出为可直接传入 Chat API 的消息列表。"""

		return Message.to_dict_list(self._messages)
