import json
from typing import Dict, List
from core.analyzer import PromptAnalyzeDimension, PromptAnalyzer, PromptAnalyzerConfig
from dotenv import load_dotenv

from core.optimizer import Optimizer, OptimizerConfig


load_dotenv()


def _get_analyze_prompt():
    with open("config/analyze_prompt.txt", "r", encoding="utf-8") as f:
        return f.read()


def _get_dimensions() -> List[PromptAnalyzeDimension]:
    with open("tmp/dimensions.json", "r", encoding="utf-8") as f:
        json_str = f.read()
        return [
            PromptAnalyzeDimension(name=item["name"], description=item["description"])
            for item in json.loads(json_str)
        ]


def _get_prompt():
    with open("tmp/prompt.txt", "r", encoding="utf-8") as f:
        return f.read()


def _get_system_prompt():
    with open("tmp/system_prompt.txt", "r", encoding="utf-8") as f:
        return f.read()


def _get_llm_response():
    with open("tmp/llm_response.txt", "r", encoding="utf-8") as f:
        return f.read()


def _get_llm_thinking():
    with open("tmp/llm_thinking.txt", "r", encoding="utf-8") as f:
        return f.read()


def _get_suggest():
    with open("tmp/suggest.txt", "r", encoding="utf-8") as f:
        return f.read()


def _get_optimize_prompt():
    with open("config/optimize_prompt.txt", "r", encoding="utf-8") as f:
        return f.read()


def analyze():
    analyzer = PromptAnalyzer(
        config=PromptAnalyzerConfig(
            model="qwen-max",
            analyze_prompt=_get_analyze_prompt(),
            dimensions=_get_dimensions(),
        )
    )
    return analyzer.analyze(
        prompt=_get_prompt(),
        llm_response=_get_llm_response(),
        llm_thinking=_get_llm_thinking(),
        suggest=_get_suggest(),
    )


def optimize(analyze_result: List[Dict], suggest: str):
    optimizer = Optimizer(
        config=OptimizerConfig(
            model="qwen-max",
            optimize_prompt=_get_optimize_prompt(),
        )
    )
    optimize_result = []
    prompt = _get_prompt()
    for item in analyze_result:
        if item["score"] != "0":
            continue
        optimize_result.extend(
            optimizer.optimize(
                prompt=prompt,
                question=f"不符合{item['name']}：{item['reason']}",
            )
        )
    if suggest:
        optimize_result.extend(
            optimizer.optimize(
                prompt=prompt,
                question=f"人工建议：{suggest}",
            )
        )
    return optimize_result


def main():
    # analyzer = PromptAnalyzer(
    #     config=PromptAnalyzerConfig(
    #         model="qwen-max",
    #         analyze_prompt=_get_analyze_prompt(),
    #         dimensions=_get_dimensions(),
    #     )
    # )
    # analyzer.analyze(
    #     prompt=_get_prompt(),
    #     llm_response=_get_llm_response(),
    #     llm_thinking=_get_llm_thinking(),
    #     suggest=_get_suggest(),
    # )
    # analyze_result = analyze()
    with open("tmp/analyze_response.json", "r", encoding="utf-8") as f:
        analyze_result = json.load(f)
    print(analyze_result)
    optimize_result = optimize(
        analyze_result=analyze_result,
        suggest=_get_suggest(),
    )
    print(optimize_result)


if __name__ == "__main__":
    main()
