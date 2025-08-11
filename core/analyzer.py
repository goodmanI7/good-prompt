from dataclasses import dataclass, field
import json
from string import Template
from typing import List, Optional

from core.llm import LLMClient, LLMConfig
from core.llm_response import get_llm_response


@dataclass
class PromptAnalyzeDimension:
    name: str
    description: str


@dataclass
class PromptAnalyzerConfig:
    model: str = "gpt-4o-mini"
    analyze_prompt: str = ""
    dimensions: List[PromptAnalyzeDimension] = field(default_factory=list)
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


class PromptAnalyzer:
    def __init__(self, config: Optional[PromptAnalyzerConfig] = None):
        self.config = config or PromptAnalyzerConfig()
        llm_config = LLMConfig(
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            frequency_penalty=self.config.frequency_penalty,
            presence_penalty=self.config.presence_penalty,
        )
        self.llm_client = LLMClient(config=llm_config)

    def analyze(
        self,
        prompt: str,
        llm_response: str,
        llm_thinking: Optional[str] = None,
        suggest: Optional[str] = None,
        **kwargs,
    ) -> str:
        dimensions = "\n".join(
            [
                f"- {dimension.name}: {dimension.description}"
                for dimension in self.config.dimensions
            ]
        )
        _context = {
            "prompt": prompt,
            "dimensions": dimensions,
            "llm_response": llm_response,
            "llm_thinking": llm_thinking,
            "suggest": suggest,
        }
        _analyze_prompt = Template(self.config.analyze_prompt).safe_substitute(_context)
        response = self.llm_client.simple_chat(
            prompt=_analyze_prompt,
        )
        return get_llm_response(response)
