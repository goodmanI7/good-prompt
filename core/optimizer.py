import json
from string import Template
from typing import Optional, List, Dict, Any
from core.llm import LLMClient, LLMConfig
from dataclasses import dataclass, field

from core.llm_response import get_llm_response


@dataclass
class OptimizerConfig:
    optimize_prompt: str = ""
    llm_config: LLMConfig = field(default_factory=LLMConfig)


@dataclass
class OptimizeResult:
    data: List[Dict[str, str]] = field(default_factory=list)
    
    def add_suggestion(self, name: str, suggest: str) -> None:
        """添加一个优化建议"""
        self.data.append({"name": name, "suggest": suggest})
    
    def get_suggestions_by_name(self, name: str) -> List[str]:
        """根据模块名称获取优化建议"""
        return [item["suggest"] for item in self.data if item["name"] == name]
    
    def get_all_names(self) -> List[str]:
        """获取所有需要优化的模块名称"""
        return list(set(item["name"] for item in self.data))
    
    def get_all_suggestions(self) -> List[str]:
        """获取所有优化建议"""
        return [item["suggest"] for item in self.data]
    
    def is_empty(self) -> bool:
        """检查是否有优化建议"""
        return len(self.data) == 0
    
    def clear(self) -> None:
        """清空所有优化建议"""
        self.data.clear()
    
    def __len__(self) -> int:
        """返回优化建议的数量"""
        return len(self.data)
    
    def __iter__(self):
        """迭代优化建议"""
        return iter(self.data)
    
    def __getitem__(self, index: int) -> Dict[str, str]:
        """通过索引获取优化建议"""
        return self.data[index]
    
    @classmethod
    def from_json(self, json_str: str) -> "OptimizeResult":
        """从JSON字符串创建优化建议"""
        data = get_llm_response(json_str)
        return OptimizeResult(data=data.get("data", []))
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "data": self.data,
        }
        
    def __str__(self) -> str:
        """字符串表示"""
        if self.is_empty():
            return "OptimizeResult(empty)"
        
        result = ""
        for i, item in enumerate(self.data, 1):
            result += f"  {i}. {item['name']}: {item['suggest']}\n"
        return result.rstrip()


class Optimizer:
    def __init__(self, config: OptimizerConfig):
        self.config = config
        self.llm_client = LLMClient(config=self.config.llm_config)
        if not self.config.optimize_prompt:
            with open("config/optimize_prompt.txt", "r", encoding="utf-8") as f:
                self.config.optimize_prompt = f.read()

    def optimize(self, prompt: str, problem: str, llm_response: str, **kwargs) -> OptimizeResult:
        _context = {
            "problem": problem,
            "prompt": prompt,
            "llm_response": llm_response,
        }
        _optimize_prompt = Template(self.config.optimize_prompt).safe_substitute(
            _context
        )
        # print(_optimize_prompt)
        response = self.llm_client.simple_chat(
            prompt=_optimize_prompt,
        )
        # print(f"Optimizer: {question}, suggest: {response}")
        result_data = get_llm_response(response)
        return OptimizeResult(data=result_data)
