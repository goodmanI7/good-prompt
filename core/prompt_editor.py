from dataclasses import dataclass, field

from core.llm import LLMClient, LLMConfig


@dataclass
class PromptEditorConfig:
    prompt_editor_prompt: str = ""
    llm_config: LLMConfig = field(default_factory=LLMConfig)


class PromptEditor:
    def __init__(self, config: PromptEditorConfig):
        self.config = config
        self.llm_client = LLMClient(config=self.config.llm_config)
        if not self.config.prompt_editor_prompt:
            with open("config/prompt_editor_prompt.txt", "r", encoding="utf-8") as f:
                self.config.prompt_editor_prompt = f.read()

    def edit(self, prompt: str, suggest: str, **kwargs) -> str:
        _context = {
            "prompt": prompt,
            "suggest": suggest,
        }
        response = self.llm_client.simple_chat(
            model=self.config.llm_config.model,
            prompt=self.config.prompt_editor_prompt.format(**_context),
        )
        return response
