from string import Template
from typing import Optional
from core.llm import LLMClient, LLMConfig
from dataclasses import dataclass

from core.llm_response import get_llm_response


@dataclass
class OptimizerConfig:
    model: str = "qwen-max"
    optimize_prompt: str = ""
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


class Optimizer:
    def __init__(self, config: OptimizerConfig):
        self.config = config
        self.llm_client = LLMClient(
            config=LLMConfig(
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        )

    def optimize(self, prompt: str, question: str, **kwargs) -> list[str]:
        _context = {
            "question": question,
            "prompt": prompt,
        }
        _optimize_prompt = Template(self.config.optimize_prompt).safe_substitute(
            _context
        )
        response = self.llm_client.simple_chat(
            prompt=_optimize_prompt,
        )
        # print(f"Optimizer: {question}, suggest: {response}")
        return get_llm_response(response)
