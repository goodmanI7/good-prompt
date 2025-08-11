from dataclasses import dataclass, field
from typing import Optional

from core.llm import LLMClient, LLMConfig


@dataclass
class GradienterConfig:
    gradient_prompt: str = ""
    llm_config: LLMConfig = field(default_factory=LLMConfig)


class Gradienter:
    def __init__(self, config: GradienterConfig):
        self.config = config
        self.llm_client = LLMClient(config=self.config.llm_config)
        if not self.config.gradient_prompt:
            with open("config/gradient_prompt.txt", "r", encoding="utf-8") as f:
                self.config.gradient_prompt = f.read()

    def gradient(self, prompt: str, suggest: str, **kwargs) -> str:
        _context = {
            "prompt": prompt,
            "suggest": suggest,
        }
        response = self.llm_client.chat_completion(
            model=self.config.llm_config.model,
            messages=[
                {
                    "role": "user",
                    "content": self.config.gradient_prompt.format(**_context),
                }
            ],
        )
        return response
