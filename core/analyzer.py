from dataclasses import dataclass, field
import json
from string import Template
from typing import List, Optional, Dict, Any

from core.llm import LLMClient, LLMConfig
from core.llm_response import get_llm_response


@dataclass
class PromptAnalyzeDimension:
    name: str
    description: str


@dataclass
class PromptAnalyzerConfig:
    analyze_prompt: str = ""
    dimensions: List[PromptAnalyzeDimension] = field(default_factory=list)
    llm_config: LLMConfig = field(default_factory=LLMConfig)


@dataclass
class DimensionScore:
    """单个维度的评分结果"""
    name: str
    score: int  # 0: 不满足, 1: 满足
    reason: str
    
    def __post_init__(self):
        """验证分数范围"""
        if self.score not in [0, 1]:
            raise ValueError(f"分数必须是0或1，当前分数: {self.score}")


@dataclass
class PromptAnalyzeResult:
    """Prompt分析结果"""
    dimension_scores: List[DimensionScore] = field(default_factory=list)
    raw_data: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_score(self, name: str, score: int, reason: str):
        """添加一个维度的评分"""
        dimension_score = DimensionScore(name=name, score=score, reason=reason)
        self.dimension_scores.append(dimension_score)
    
    def get_score_by_name(self, name: str) -> Optional[DimensionScore]:
        """根据维度名称获取评分"""
        for score in self.dimension_scores:
            if score.name == name:
                return score
        return None
    
    def get_total_score(self) -> int:
        """获取总分"""
        return sum(score.score for score in self.dimension_scores)
    
    def get_score_percentage(self) -> float:
        """获取得分百分比"""
        if not self.dimension_scores:
            return 0.0
        return (self.get_total_score() / len(self.dimension_scores)) * 100
    
    def get_failed_dimensions(self) -> List[DimensionScore]:
        """获取未满足的维度"""
        return [score for score in self.dimension_scores if score.score == 0]
    
    def get_passed_dimensions(self) -> List[DimensionScore]:
        """获取满足的维度"""
        return [score for score in self.dimension_scores if score.score == 1]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "dimension_scores": [
                {
                    "name": score.name,
                    "score": score.score,
                    "reason": score.reason
                }
                for score in self.dimension_scores
            ],
            "total_score": self.get_total_score(),
            "score_percentage": self.get_score_percentage(),
            "total_dimensions": len(self.dimension_scores),
            "passed_dimensions": len(self.get_passed_dimensions()),
            "failed_dimensions": len(self.get_failed_dimensions())
        }
    
    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'PromptAnalyzeResult':
        """从JSON字符串创建结果对象"""
        try:
            data = get_llm_response(json_str)
            result = cls()
            
            if isinstance(data, list):
                # 处理原始格式的JSON数组
                for item in data:
                    if isinstance(item, dict) and 'name' in item and 'score' in item and 'reason' in item:
                        result.add_score(
                            name=item['name'],
                            score=int(item['score']),
                            reason=item['reason']
                        )
            elif isinstance(data, dict) and 'dimension_scores' in data:
                # 处理完整格式的JSON对象
                for score_data in data['dimension_scores']:
                    result.add_score(
                        name=score_data['name'],
                        score=int(score_data['score']),
                        reason=score_data['reason']
                    )
            
            return result
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise ValueError(f"无法解析JSON数据: {e}")
    
    def __str__(self) -> str:
        """字符串表示"""
        if not self.dimension_scores:
            return "PromptAnalyzeResult(无评分数据)"
        
        lines = [f"PromptAnalyzeResult(总分: {self.get_total_score()}/{len(self.dimension_scores)}, 得分率: {self.get_score_percentage():.1f}%)"]
        for score in self.dimension_scores:
            status = "✓" if score.score == 1 else "✗"
            lines.append(f"  {status} {score.name}: {score.reason}")
        return "\n".join(lines)


class PromptAnalyzer:
    def __init__(self, config: Optional[PromptAnalyzerConfig] = None):
        self.config = config or PromptAnalyzerConfig()
        self.llm_client = LLMClient(config=self.config.llm_config)
        if not self.config.analyze_prompt:
            with open("config/analyze_prompt.txt", "r", encoding="utf-8") as f:
                self.config.analyze_prompt = f.read()
            

    def analyze(
        self,
        prompt: str,
        llm_response: str,
        llm_thinking: Optional[str] = None,
        suggest: Optional[str] = None,
        **kwargs,
    ) -> PromptAnalyzeResult:
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
        
        # 解析LLM返回的JSON结果
        try:
            return PromptAnalyzeResult.from_json(response)
        except ValueError as e:
            # 如果解析失败，返回空结果
            print(f"警告: 无法解析LLM返回的结果: {e}")
            return PromptAnalyzeResult()
