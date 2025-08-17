import json
import os
from string import Template
from typing import Dict, List
from core.analyzer import (
    PromptAnalyzeDimension,
    PromptAnalyzeResult,
    PromptAnalyzer,
    PromptAnalyzerConfig,
)
from dotenv import load_dotenv

from core.llm import LLMConfig
from core.optimizer import OptimizeResult, Optimizer, OptimizerConfig
from core.prompt_editor import PromptEditor, PromptEditorConfig


load_dotenv()


def _get_dimensions(scene: str) -> List[PromptAnalyzeDimension]:
    with open(f"tmp/{scene}/dimensions.json", "r", encoding="utf-8") as f:
        json_str = f.read()
        return [
            PromptAnalyzeDimension(name=item["name"], description=item["description"])
            for item in json.loads(json_str)
        ]


def _get_prompt(scene: str):
    with open(f"tmp/{scene}/system_prompt.txt", "r", encoding="utf-8") as f:
        return f.read()


def _get_content(scene: str):
    with open(f"tmp/{scene}/content.txt", "r", encoding="utf-8") as f:
        return f.read()


def _get_system_prompt():
    with open("tmp/system_prompt.txt", "r", encoding="utf-8") as f:
        return f.read()


def _get_llm_response(scene: str):
    with open(f"tmp/{scene}/llm_response.txt", "r", encoding="utf-8") as f:
        return f.read()


def _get_llm_thinking(scene: str):
    with open(f"tmp/{scene}/llm_thinking.txt", "r", encoding="utf-8") as f:
        return f.read()


def _get_suggest(scene: str):
    with open(f"tmp/{scene}/suggest.txt", "r", encoding="utf-8") as f:
        return f.read()


def _get_optimize_prompt():
    with open("config/optimize_prompt.txt", "r", encoding="utf-8") as f:
        return f.read()


def analyze(scene: str, use_cache: bool = False):
    if use_cache and os.path.exists(f"tmp/{scene}/cache/analyze_response.json"):
        with open(
            f"tmp/{scene}/cache/analyze_response.json", "r", encoding="utf-8"
        ) as f:
            analyze_str = f.read()
            if analyze_str:
                return PromptAnalyzeResult.from_json(analyze_str)

    analyzer = PromptAnalyzer(
        config=PromptAnalyzerConfig(
            llm_config=LLMConfig(
                model="qwen-max",
            ),
            dimensions=_get_dimensions(scene),
        )
    )
    result = analyzer.analyze(
        prompt=_get_prompt(scene),
        llm_response=_get_llm_response(scene),
        llm_thinking=_get_llm_thinking(scene),
        suggest=_get_suggest(scene),
    )

    with open(f"tmp/{scene}/cache/analyze_response.json", "w", encoding="utf-8") as f:
        f.write(result.to_json())
    return result


def optimize(
    analyze_result: PromptAnalyzeResult,
    suggest: str,
    scene: str,
    use_cache: bool = False,
):
    if use_cache and os.path.exists(f"tmp/{scene}/cache/optimize_response.json"):
        with open(
            f"tmp/{scene}/cache/optimize_response.json", "r", encoding="utf-8"
        ) as f:
            optimize_str = f.read()
            if optimize_str:
                return OptimizeResult.from_json(optimize_str)
    optimizer = Optimizer(
        config=OptimizerConfig(
            llm_config=LLMConfig(
                model="qwen-max",
            ),
        )
    )
    problem_list = []
    prompt = _get_prompt(scene)
    llm_response = _get_llm_response(scene)
    optimize_result = OptimizeResult()
    for item in analyze_result.dimension_scores:
        if item.score != 0:
            continue
        problem_list.append(f"{item.name}：{item.reason}")
    if suggest:
        problem_list.append(f"人工建议：{suggest}")
    for problem in problem_list:
        _optimize_result = optimizer.optimize(
            prompt=prompt,
            problem=problem,
            llm_response=llm_response,
        )
        optimize_result.data.extend(_optimize_result.data)
    with open(f"tmp/{scene}/cache/optimize_response.json", "w", encoding="utf-8") as f:
        f.write(optimize_result.to_json())
    return optimize_result


def prompt_editor(prompt: str, suggest: str):
    prompt_editor = PromptEditor(
        config=PromptEditorConfig(
            llm_config=LLMConfig(model="qwen-max"),
        )
    )
    return prompt_editor.edit(prompt=prompt, suggest=suggest)


def main(scene: str, use_cache: bool = False):
    analyze_result = analyze(scene, use_cache)
    print(analyze_result)
    optimize_result = optimize(
        analyze_result, suggest=_get_suggest(scene), scene=scene, use_cache=use_cache
    )
    print("-" * 100)
    print(optimize_result)
    print("-" * 100)
    prompt_editor_result = prompt_editor(
        prompt=_get_prompt(scene),
        suggest=optimize_result.__str__(),
    )
    print(prompt_editor_result)
    print("-" * 100)
    with open(f"tmp/{scene}/system_prompt.txt", "w", encoding="utf-8") as f:
        f.write(prompt_editor_result)
    template = Template(prompt_editor_result)
    print(template.substitute(content=_get_content(scene)))
    print("-" * 100)


if __name__ == "__main__":
    main(scene="novel", use_cache=False)
    # main(scene="jianli", use_cache=True)
