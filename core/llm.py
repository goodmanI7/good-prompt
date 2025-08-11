import os
import json
import logging
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
import openai
from openai import OpenAI

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """LLM配置类"""

    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    timeout: int = 60
    api_key: Optional[str] = None
    base_url: Optional[str] = None


class LLMClient:
    """LLM客户端类，用于调用OpenAI API"""

    def __init__(self, config: Optional[LLMConfig] = None):
        """
        初始化LLM客户端

        Args:
            config: LLM配置，如果为None则使用默认配置
        """
        self.config = config or LLMConfig()
        self._setup_client()

    def _setup_client(self):
        """设置OpenAI客户端"""
        # 获取API密钥
        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API密钥未设置，请设置OPENAI_API_KEY环境变量或传入config"
            )

        # 创建客户端
        client_kwargs = {"api_key": api_key}
        client_kwargs["base_url"] = self.config.base_url or os.getenv("OPENAI_BASE_URL")
        
        # 配置http不走代理和超时时间
        import httpx
        client_kwargs["http_client"] = httpx.Client(
            proxy=None,  # None表示不使用代理
            timeout=600.0  # 10分钟超时时间
        )

        self.client = OpenAI(**client_kwargs)

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        调用聊天完成API

        Args:
            messages: 消息列表，格式为[{"role": "user", "content": "..."}]
            model: 模型名称，如果为None则使用配置中的模型
            temperature: 温度参数，如果为None则使用配置中的温度
            max_tokens: 最大token数，如果为None则使用配置中的值
            **kwargs: 其他参数

        Returns:
            API响应字典
        """
        try:
            # 使用传入的参数或默认配置
            model = model or self.config.model
            temperature = (
                temperature if temperature is not None else self.config.temperature
            )
            max_tokens = (
                max_tokens if max_tokens is not None else self.config.max_tokens
            )

            # 构建请求参数
            request_params = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "top_p": self.config.top_p,
                "frequency_penalty": self.config.frequency_penalty,
                "presence_penalty": self.config.presence_penalty,
                "timeout": self.config.timeout,
                **kwargs,
            }

            # 移除None值
            request_params = {k: v for k, v in request_params.items() if v is not None}

            logger.info(f"调用OpenAI API，模型: {model}")
            response = self.client.chat.completions.create(**request_params)

            return {
                "content": response.choices[0].message.content,
                "model": response.model,
                "usage": response.usage.dict() if response.usage else None,
                "finish_reason": response.choices[0].finish_reason,
                "raw_response": response,
            }

        except Exception as e:
            logger.error(f"调用OpenAI API失败: {str(e)}")
            raise

    def simple_chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> str:
        """
        简单的聊天接口

        Args:
            prompt: 用户输入
            system_prompt: 系统提示，可选
            model: 模型名称，可选
            temperature: 温度参数，可选
            **kwargs: 其他参数

        Returns:
            模型回复内容
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        response = self.chat_completion(
            messages=messages, model=model, temperature=temperature, **kwargs
        )

        return response["content"]

    def format_prompt(self, template: str, **kwargs) -> str:
        """
        格式化提示模板

        Args:
            template: 提示模板
            **kwargs: 模板参数

        Returns:
            格式化后的提示
        """
        try:
            return template.format(**kwargs)
        except KeyError as e:
            logger.error(f"提示模板格式化失败，缺少参数: {e}")
            raise

    def batch_chat(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> List[str]:
        """
        批量处理多个提示

        Args:
            prompts: 提示列表
            system_prompt: 系统提示，可选
            model: 模型名称，可选
            temperature: 温度参数，可选
            **kwargs: 其他参数

        Returns:
            回复列表
        """
        results = []

        for i, prompt in enumerate(prompts):
            try:
                logger.info(f"处理第 {i+1}/{len(prompts)} 个提示")
                response = self.simple_chat(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    model=model,
                    temperature=temperature,
                    **kwargs,
                )
                results.append(response)
            except Exception as e:
                logger.error(f"处理第 {i+1} 个提示失败: {str(e)}")
                results.append(f"错误: {str(e)}")

        return results


# 便捷函数
def create_llm_client(
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    api_key: Optional[str] = None,
    **kwargs,
) -> LLMClient:
    """
    创建LLM客户端的便捷函数

    Args:
        model: 模型名称
        temperature: 温度参数
        api_key: API密钥
        **kwargs: 其他配置参数

    Returns:
        LLMClient实例
    """
    config = LLMConfig(model=model, temperature=temperature, api_key=api_key, **kwargs)
    return LLMClient(config)


def chat_with_llm(
    prompt: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    system_prompt: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs,
) -> str:
    """
    与LLM聊天的便捷函数

    Args:
        prompt: 用户输入
        model: 模型名称
        temperature: 温度参数
        system_prompt: 系统提示
        api_key: API密钥
        **kwargs: 其他参数

    Returns:
        模型回复
    """
    client = create_llm_client(
        model=model, temperature=temperature, api_key=api_key, **kwargs
    )
    return client.simple_chat(
        prompt=prompt, system_prompt=system_prompt, model=model, temperature=temperature
    )
