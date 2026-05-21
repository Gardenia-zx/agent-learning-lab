from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.message import MessageBuilder


def test_message_builder_builds_correct_messages():
    messages = (
        MessageBuilder()
        .system("你是一个严谨的助手")
        .user("解释什么是 Agent")
        .build()
    )

    assert messages == [
        {"role": "system", "content": "你是一个严谨的助手"},
        {"role": "user", "content": "解释什么是 Agent"},
    ]
