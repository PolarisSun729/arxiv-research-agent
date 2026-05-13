import os
import json
import re
from datetime import datetime
from typing import List, Dict, Optional, Iterator, Any
import logging
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from openai import OpenAI
import requests
from utils.model_utils import get_huggingface_model_path

# 设置环境变量以启用 Apple Silicon (MPS) 回退到 CPU (当遇到不支持的操作时会自动回退到 CPU 执行)
# 目前 PyTorch 版本 ≥ 1.13 时，才支持 Apple 的 Metal Performance Shaders (MPS) ，而且暂不支持「多 GPU」，另外，部分训练操作尚未完全实现
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

logger = logging.getLogger(__name__)

# 阿里云百炼 / 通义千问 API Key。
# 按你的要求这里直接写在代码里；请替换为你自己的真实 Key。
QWEN_API_KEY = "<REMOVED_API_KEY>"
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL_NAME = "qwen3.6-plus"

class GenerationService:
    """
    生成服务类：负责调用不同的模型提供商（HuggingFace、OpenAI、DeepSeek）生成回答
    支持本地模型和API调用，并将生成结果保存到文件
    """
    def __init__(self):
        """
        初始化生成服务，配置支持的模型列表和创建输出目录
        """
        self.models = {
            "huggingface": {
                "Llama-2-7b-chat": "meta-llama/Llama-2-7b-chat-hf",
                "DeepSeek-7b": "deepseek-ai/deepseek-llm-7b-chat",
                "DeepSeek-R1-Distill-Qwen": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
            },
            "openai": {
                "gpt-3.5-turbo": "gpt-3.5-turbo",
                "gpt-4": "gpt-4",
            },
            "qwen": {
                "qwen3.6-plus": "qwen3.6-plus",
            },
            "deepseek": {
                "deepseek-v3": "deepseek-chat",
                "deepseek-r1": "deepseek-reasoner",
            }
        }
        
        # 确保输出目录存在
        os.makedirs("05-generation-results", exist_ok=True)
        
    def _load_huggingface_model(self, model_name: str):
        """
        加载HuggingFace模型
        
        参数:
            model_name: 模型名称，对应self.models["huggingface"]中的键
            
        返回:
            model: 加载的模型
            tokenizer: 对应的分词器
        """
        try:
            model_name = self.models["huggingface"][model_name]
            model_name = get_huggingface_model_path(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
            )
            return model, tokenizer
        except Exception as e:
            logger.error(f"Error loading HuggingFace model: {str(e)}")
            raise

    def _generate_with_huggingface(
        self,
        model_name: str,
        query: str,
        context: str,
        max_length: int = 512
    ) -> str:
        """
        使用HuggingFace模型生成回答
        
        参数:
            model_name: 模型名称
            query: 用户查询
            context: 上下文信息
            max_length: 生成文本的最大长度
            
        返回:
            生成的回答文本
        """
        try:
            model, tokenizer = self._load_huggingface_model(model_name)
            
            # 构建提示
            prompt = f"""请基于以下上下文回答问题。如果上下文中没有相关信息，请说明无法回答。

                        问题：{query}

                        上下文：
                        {context}

                        回答："""
        
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True
            )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response.split("回答：")[-1].strip()
            
        except Exception as e:
            logger.error(f"Error generating with HuggingFace: {str(e)}")
            raise

    def _generate_with_openai(
        self,
        model_name: str,
        query: str,
        context: str,
        api_key: Optional[str] = None
    ) -> str:
        """
        使用OpenAI API生成回答
        
        参数:
            model_name: 模型名称
            query: 用户查询
            context: 上下文信息
            api_key: OpenAI API密钥，如不提供则从环境变量获取
            
        返回:
            生成的回答文本
        """
        try:
            if not api_key:
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OpenAI API key not provided")
                    
            client = OpenAI(api_key=api_key)
            
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer the question."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
            ]
            
            response = client.chat.completions.create(
                model=self.models["openai"][model_name],
                messages=messages,
                temperature=0.7,
                max_tokens=512
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating with OpenAI: {str(e)}")
            raise

    def _generate_with_qwen_responses(
        self,
        query: str,
        context: str,
        api_key: Optional[str] = None,
        model_name: str = QWEN_MODEL_NAME,
    ) -> str:
        """
        使用阿里云百炼的 OpenAI 兼容 Responses API 生成答案。

        这里采用 Qwen3.6-Plus，输入为检索到的上下文 + 问题。
        """
        try:
            if not api_key:
                api_key = QWEN_API_KEY
            if not api_key or api_key == "PASTE_YOUR_QWEN_API_KEY_HERE":
                raise ValueError("Qwen API key not provided")

            client = OpenAI(
                api_key=api_key,
                base_url=QWEN_BASE_URL,
            )

            prompt = (
                "你是一个严谨的论文问答助手。"
                "请仅根据给定的论文上下文回答问题；"
                "如果上下文中没有足够信息，请明确说明无法从当前论文内容中确定。"
                "\n\n论文上下文：\n"
                f"{context}\n\n"
                f"问题：{query}\n\n"
                "回答："
            )

            response = client.responses.create(
                model=model_name,
                input=prompt,
            )

            answer = getattr(response, "output_text", None)
            if answer:
                return answer.strip()

            # 兜底解析，防止 SDK 返回结构变化时拿不到 output_text。
            output_parts = []
            for item in getattr(response, "output", []) or []:
                if getattr(item, "type", None) == "message":
                    for content in getattr(item, "content", []) or []:
                        text = getattr(content, "text", None)
                        if text:
                            output_parts.append(text)
            if output_parts:
                return "".join(output_parts).strip()

            raise ValueError("Qwen response did not contain output text")

        except Exception as e:
            logger.error(f"Error generating with Qwen Responses API: {str(e)}")
            raise

    def complete_with_qwen(
        self,
        prompt: str,
        api_key: Optional[str] = None,
        model_name: str = QWEN_MODEL_NAME,
    ) -> str:
        try:
            if not api_key:
                api_key = QWEN_API_KEY
            if not api_key:
                raise ValueError("Qwen API key not provided")

            client = OpenAI(
                api_key=api_key,
                base_url=QWEN_BASE_URL,
            )
            response = client.responses.create(
                model=model_name,
                input=prompt,
            )

            answer = getattr(response, "output_text", None)
            if answer:
                return answer.strip()

            output_parts = []
            for item in getattr(response, "output", []) or []:
                if getattr(item, "type", None) == "message":
                    for content in getattr(item, "content", []) or []:
                        text = getattr(content, "text", None)
                        if text:
                            output_parts.append(text)
            if output_parts:
                return "".join(output_parts).strip()

            raise ValueError("Qwen response did not contain output text")
        except Exception as e:
            logger.error(f"Error completing prompt with Qwen: {str(e)}")
            raise

    def rewrite_query_for_retrieval(
        self,
        question: str,
        max_queries: int = 3,
        api_key: Optional[str] = None,
        model_name: str = QWEN_MODEL_NAME,
    ) -> List[str]:
        prompt = (
            "You are helping a retrieval system search an English academic paper.\n"
            "Rewrite the user's question into concise retrieval-oriented English queries.\n"
            "Rules:\n"
            "1. Return JSON only.\n"
            "2. Output format: {\"queries\": [\"...\", \"...\"]}\n"
            "3. Keep each query short and retrieval-friendly.\n"
            "4. Preserve the user's intent.\n"
            "5. Focus on terminology likely to appear in a research paper.\n"
            f"6. Return at most {max_queries} rewritten queries.\n\n"
            f"User question: {question}"
        )
        response = self.complete_with_qwen(prompt, api_key=api_key, model_name=model_name)
        data = json.loads(self._extract_json_block(response))
        queries = data.get("queries", [])
        return [str(query).strip() for query in queries if str(query).strip()]

    def generate_hyde_document(
        self,
        question: str,
        api_key: Optional[str] = None,
        model_name: str = QWEN_MODEL_NAME,
    ) -> str:
        prompt = (
            "You are writing a hypothetical passage to help semantic retrieval over a single English research paper.\n"
            "Write one short paragraph that likely resembles a relevant paper chunk answering the question.\n"
            "Rules:\n"
            "1. Use English.\n"
            "2. Do not mention that the passage is hypothetical.\n"
            "3. Keep it under 120 words.\n"
            "4. Focus on paper-style terminology, section wording, and likely evidence.\n\n"
            f"Question: {question}"
        )
        return self.complete_with_qwen(prompt, api_key=api_key, model_name=model_name).strip()

    def _extract_json_block(self, text: str) -> str:
        fenced_match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fenced_match:
            return fenced_match.group(1)

        plain_match = re.search(r"(\{.*\})", text, re.DOTALL)
        if plain_match:
            return plain_match.group(1)

        return text

    def _build_qwen_prompt(self, query: str, context: str) -> str:
        return (
            "你是一个严谨的论文问答助手。"
            "请仅根据给定的论文上下文回答问题；"
            "如果上下文中没有足够信息，请明确说明无法从当前论文内容中确定。"
            "\n\n论文上下文：\n"
            f"{context}\n\n"
            f"问题：{query}\n\n"
            "回答："
        )

    def stream_qwen_responses(
        self,
        query: str,
        context: str,
        api_key: Optional[str] = None,
        model_name: str = QWEN_MODEL_NAME,
    ) -> Iterator[Dict[str, Any]]:
        """
        使用 Qwen Responses API 的流式输出，逐段返回增量文本。
        """
        try:
            if not api_key:
                api_key = QWEN_API_KEY
            if not api_key:
                raise ValueError("Qwen API key not provided")

            client = OpenAI(
                api_key=api_key,
                base_url=QWEN_BASE_URL,
            )

            stream = client.responses.create(
                model=model_name,
                input=self._build_qwen_prompt(query, context),
                stream=True,
            )

            answer_parts: List[str] = []
            for event in stream:
                event_type = getattr(event, "type", "")
                if event_type == "response.output_text.delta":
                    delta = getattr(event, "delta", "") or ""
                    if delta:
                        answer_parts.append(delta)
                        yield {"type": "delta", "delta": delta}
                elif event_type == "response.completed":
                    response = getattr(event, "response", None)
                    usage = getattr(response, "usage", None) if response else None
                    yield {
                        "type": "completed",
                        "answer": "".join(answer_parts),
                        "usage": {
                            "input_tokens": getattr(usage, "input_tokens", None) if usage else None,
                            "output_tokens": getattr(usage, "output_tokens", None) if usage else None,
                            "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
                        },
                    }
                    return

            yield {
                "type": "completed",
                "answer": "".join(answer_parts),
                "usage": None,
            }

        except Exception as e:
            logger.error(f"Error streaming with Qwen Responses API: {str(e)}")
            raise

    def _generate_with_deepseek(
        self,
        model_name: str,
        query: str,
        context: str,
        api_key: Optional[str] = None,
        show_reasoning: bool = True
    ) -> str:
        """
        使用DeepSeek API生成回答
        
        参数:
            model_name: 模型名称
            query: 用户查询
            context: 上下文信息
            api_key: DeepSeek API密钥，如不提供则从环境变量获取
            show_reasoning: 是否显示推理过程（仅对推理模型有效）
            
        返回:
            生成的回答文本，对于推理模型可能包含思维过程
        """
        try:
            if not api_key:
                api_key = os.getenv("DEEPSEEK_API_KEY")
                if not api_key:
                    raise ValueError("DeepSeek API key not provided")
                    
            client = OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com"
            )
            
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer the question."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
            ]
            
            response = client.chat.completions.create(
                model=self.models["deepseek"][model_name],
                messages=messages,
                max_tokens=512,
                stream=False
            )
            
            # 如果是推理模型，处理思维链输出
            if model_name == "deepseek-r1":
                message = response.choices[0].message
                reasoning = message.reasoning_content
                answer = message.content
                
                if show_reasoning and reasoning:
                    return f"【思维过程】\n{reasoning}\n\n【最终答案】\n{answer}"
                return answer
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating with DeepSeek: {str(e)}")
            raise

    def generate(
        self,
        provider: str,
        model_name: str,
        query: str,
        search_results: List[Dict],
        api_key: Optional[str] = None,
        show_reasoning: bool = True
    ) -> Dict:
        """
        生成回答并保存结果
        
        参数:
            provider: 模型提供商，可选值为"huggingface"、"openai"、"deepseek"
            model_name: 模型名称
            query: 用户查询
            search_results: 搜索结果列表，用于构建上下文
            api_key: API密钥（对于API调用）
            show_reasoning: 是否显示推理过程（仅对DeepSeek推理模型有效）
            
        返回:
            包含生成回答和保存路径的字典
        """
        try:
            # 准备上下文
            context = "\n\n".join([
                f"[Source {i+1}]: {result['text']}"
                for i, result in enumerate(search_results)
            ])
            
            # 根据不同提供商生成回答
            if provider == "huggingface":
                response = self._generate_with_huggingface(model_name, query, context)
            elif provider == "openai":
                response = self._generate_with_openai(model_name, query, context, api_key)
            elif provider == "qwen":
                response = self._generate_with_qwen_responses(query, context, api_key, model_name)
            elif provider == "deepseek":
                response = self._generate_with_deepseek(model_name, query, context, api_key, show_reasoning)
            else:
                raise ValueError(f"Unsupported provider: {provider}")
                
            # 准备保存的结果
            result = {
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "provider": provider,
                "model": model_name,
                "response": response,
                "context": search_results
            }
            
            # 生成文件名并保存
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"generation_{provider}_{model_name}_{timestamp}.json"
            filepath = os.path.join("05-generation-results", filename)
            
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
                
            return {
                "response": response,
                "saved_filepath": filepath
            }
            
        except Exception as e:
            logger.error(f"Error in generation: {str(e)}")
            raise

    def get_available_models(self) -> Dict:
        """
        获取可用的模型列表
        
        返回:
            包含所有支持模型的字典
        """
        return self.models 
