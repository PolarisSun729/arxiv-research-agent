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
from utils.config import GENERATION_CONFIG
# 璁剧疆鐜鍙橀噺浠ュ惎鐢?Apple Silicon (MPS) 鍥為€€鍒?CPU (褰撻亣鍒颁笉鏀寔鐨勬搷浣滄椂浼氳嚜鍔ㄥ洖閫€鍒?CPU 鎵ц)
# 鐩墠 PyTorch 鐗堟湰 鈮?1.13 鏃讹紝鎵嶆敮鎸?Apple 鐨?Metal Performance Shaders (MPS) 锛岃€屼笖鏆備笉鏀寔銆屽 GPU銆嶏紝鍙﹀锛岄儴鍒嗚缁冩搷浣滃皻鏈畬鍏ㄥ疄鐜?
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

logger = logging.getLogger(__name__)

# 闃块噷浜戠櫨鐐?/ 閫氫箟鍗冮棶 API Key銆?
# 鎸変綘鐨勮姹傝繖閲岀洿鎺ュ啓鍦ㄤ唬鐮侀噷锛涜鏇挎崲涓轰綘鑷繁鐨勭湡瀹?Key銆?
QWEN_API_KEY = GENERATION_CONFIG["qwen_api_key"]
QWEN_BASE_URL = GENERATION_CONFIG["qwen_base_url"]
QWEN_MODEL_NAME = GENERATION_CONFIG["qwen_model_name"]

class GenerationService:
    """
    鐢熸垚鏈嶅姟绫伙細璐熻矗璋冪敤涓嶅悓鐨勬ā鍨嬫彁渚涘晢锛圚uggingFace銆丱penAI銆丏eepSeek锛夌敓鎴愬洖绛?
    鏀寔鏈湴妯″瀷鍜孉PI璋冪敤锛屽苟灏嗙敓鎴愮粨鏋滀繚瀛樺埌鏂囦欢
    """
    def __init__(self):
        """
        鍒濆鍖栫敓鎴愭湇鍔★紝閰嶇疆鏀寔鐨勬ā鍨嬪垪琛ㄥ拰鍒涘缓杈撳嚭鐩綍
        """
        self.models = {
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
        
        # 纭繚杈撳嚭鐩綍瀛樺湪
        os.makedirs("05-generation-results", exist_ok=True)
        
    def _load_huggingface_model(self, model_name: str):
        """
        鍔犺浇HuggingFace妯″瀷
        
        鍙傛暟:
            model_name: 妯″瀷鍚嶇О锛屽搴攕elf.models["huggingface"]涓殑閿?
            
        杩斿洖:
            model: 鍔犺浇鐨勬ā鍨?
            tokenizer: 瀵瑰簲鐨勫垎璇嶅櫒
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
        浣跨敤HuggingFace妯″瀷鐢熸垚鍥炵瓟
        
        鍙傛暟:
            model_name: 妯″瀷鍚嶇О
            query: 鐢ㄦ埛鏌ヨ
            context: 涓婁笅鏂囦俊鎭?
            max_length: 鐢熸垚鏂囨湰鐨勬渶澶ч暱搴?
            
        杩斿洖:
            鐢熸垚鐨勫洖绛旀枃鏈?
        """
        try:
            model, tokenizer = self._load_huggingface_model(model_name)
            
            prompt = f"""Answer the question strictly based on the provided context.
If the context does not contain enough information, say you cannot determine it.

Question: {query}

Context:
{context}

Answer:"""
        
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True
            )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response.split("Answer:")[-1].strip()
            
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
        浣跨敤OpenAI API鐢熸垚鍥炵瓟
        
        鍙傛暟:
            model_name: 妯″瀷鍚嶇О
            query: 鐢ㄦ埛鏌ヨ
            context: 涓婁笅鏂囦俊鎭?
            api_key: OpenAI API瀵嗛挜锛屽涓嶆彁渚涘垯浠庣幆澧冨彉閲忚幏鍙?
            
        杩斿洖:
            鐢熸垚鐨勫洖绛旀枃鏈?
        """
        try:
            if not api_key:
                api_key = GENERATION_CONFIG["openai_api_key"]
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
        浣跨敤闃块噷浜戠櫨鐐肩殑 OpenAI 鍏煎 Responses API 鐢熸垚绛旀銆?

        杩欓噷閲囩敤 Qwen3.6-Plus锛岃緭鍏ヤ负妫€绱㈠埌鐨勪笂涓嬫枃 + 闂銆?
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

            prompt = (
                "You are a strict academic QA assistant. Answer only from the provided context.\n"
                "If the context is insufficient, say you cannot determine it.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {query}\n\n"
                "Answer:"
            )

            response = client.responses.create(
                model=model_name,
                input=prompt,
            )

            answer = getattr(response, "output_text", None)
            if answer:
                return answer.strip()

            # 鍏滃簳瑙ｆ瀽锛岄槻姝?SDK 杩斿洖缁撴瀯鍙樺寲鏃舵嬁涓嶅埌 output_text銆?
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
        paper_context: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        model_name: str = QWEN_MODEL_NAME,
    ) -> List[str]:
        data = self.plan_queries_for_retrieval(
            question=question,
            max_queries=max_queries,
            paper_context=paper_context,
            api_key=api_key,
            model_name=model_name,
        )
        queries = data.get("rewrite_queries", [])
        normalized_queries: List[str] = []
        for item in queries:
            if isinstance(item, dict):
                query = str(item.get("query", "")).strip()
                if query:
                    normalized_queries.append(query)
            elif isinstance(item, str) and item.strip():
                normalized_queries.append(item.strip())
        return normalized_queries

    def plan_queries_for_retrieval(
        self,
        question: str,
        max_queries: int = 5,
        paper_context: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        model_name: str = QWEN_MODEL_NAME,
    ) -> Dict[str, Any]:
        paper_context = paper_context or {}
        title = str(paper_context.get("title", "") or "").strip()
        abstract = str(paper_context.get("abstract", "") or "").strip()
        section_titles = [str(item).strip() for item in (paper_context.get("section_titles", []) or []) if str(item).strip()]
        candidate_terms = [str(item).strip() for item in (paper_context.get("candidate_terms", []) or []) if str(item).strip()]

        prompt = (
            "You are a query planner for retrieval over a single academic paper.\n"
            "Classify the user's question, infer the user's intent, and produce retrieval queries grounded in the paper context.\n"
            "Rules:\n"
            "1. Return JSON only.\n"
            "2. Do not invent paper-specific names, datasets, modules, or methods.\n"
            "3. Prefer terms copied from the paper title, abstract, and section titles.\n"
            "4. Keep each query short and retrieval-friendly.\n"
            "5. Generate 3 to 5 queries with different retrieval angles, not paraphrase duplicates.\n"
            "6. Prefer English retrieval queries unless the paper context is clearly Chinese.\n"
            "7. Use one of these question types only: method_flow, experiment_setup, results_analysis, contribution, limitation, dataset, metric, figure_table, summary, other.\n"
            "8. The JSON schema must be:\n"
            "   {\"question_type\": \"...\", \"intent_summary\": \"...\", \"paper_terms\": [\"...\"], \"preferred_sections\": [\"...\"], \"rewrite_queries\": [{\"query\": \"...\", \"focus\": \"...\", \"channels\": [\"vector\", \"keyword\"]}]}\n"
            f"9. Return at most {max_queries} rewrite queries.\n\n"
            f"User question: {question}\n\n"
            f"Paper title: {title or 'N/A'}\n"
            f"Paper abstract: {abstract[:1800] or 'N/A'}\n"
            f"Section titles: {', '.join(section_titles[:16]) or 'N/A'}\n"
            f"Candidate paper terms: {', '.join(candidate_terms[:24]) or 'N/A'}"
        )
        response = self.complete_with_qwen(prompt, api_key=api_key, model_name=model_name)
        data = json.loads(self._extract_json_block(response))
        return self._normalize_query_plan(data, question, max_queries=max_queries, paper_context=paper_context)

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

    def _normalize_query_plan(
        self,
        data: Dict[str, Any],
        question: str,
        max_queries: int = 5,
        paper_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        paper_context = paper_context or {}
        paper_terms = [str(item).strip() for item in (data.get("paper_terms", []) or []) if str(item).strip()]
        preferred_sections = [str(item).strip() for item in (data.get("preferred_sections", []) or []) if str(item).strip()]
        paper_terms = list(dict.fromkeys(paper_terms))
        preferred_sections = list(dict.fromkeys(preferred_sections))

        rewrite_queries: List[Dict[str, Any]] = []
        raw_queries = data.get("rewrite_queries")
        if isinstance(raw_queries, list):
            for item in raw_queries:
                if isinstance(item, dict):
                    query = str(item.get("query", "")).strip()
                    if not query:
                        continue
                    rewrite_queries.append(
                        {
                            "query": query,
                            "focus": str(item.get("focus", "")).strip(),
                            "channels": [
                                str(channel).strip()
                                for channel in (item.get("channels", []) or [])
                                if str(channel).strip()
                            ]
                            or ["vector", "keyword"],
                        }
                    )
                elif isinstance(item, str) and item.strip():
                    rewrite_queries.append(
                        {
                            "query": item.strip(),
                            "focus": "",
                            "channels": ["vector", "keyword"],
                        }
                    )
        else:
            legacy_queries = data.get("queries", [])
            if isinstance(legacy_queries, list):
                for item in legacy_queries:
                    if isinstance(item, str) and item.strip():
                        rewrite_queries.append(
                            {
                                "query": item.strip(),
                                "focus": "",
                                "channels": ["vector", "keyword"],
                            }
                        )

        if len(rewrite_queries) > max_queries:
            rewrite_queries = rewrite_queries[:max_queries]

        if not rewrite_queries:
            fallback_terms = paper_terms[:4]
            if not fallback_terms:
                fallback_terms = self._extract_fallback_terms(
                    " ".join(
                        [
                            str(paper_context.get("title", "") or ""),
                            str(paper_context.get("abstract", "") or ""),
                            " ".join(paper_context.get("section_titles", []) or []),
                            question,
                        ]
                    )
                )
            fallback_query = " ".join(fallback_terms[:6]).strip() or question.strip()
            rewrite_queries = [
                {
                    "query": fallback_query,
                    "focus": "semantic",
                    "channels": ["vector", "keyword"],
                }
            ]
        elif len(rewrite_queries) < 3:
            question_type = str(data.get("question_type", "other")).strip() or "other"
            fallback_terms = paper_terms[:4]
            if not fallback_terms:
                fallback_terms = self._extract_fallback_terms(
                    " ".join(
                        [
                            str(paper_context.get("title", "") or ""),
                            str(paper_context.get("abstract", "") or ""),
                            " ".join(paper_context.get("section_titles", []) or []),
                            question,
                        ]
                    )
                )

            def build_extra_query(extra_focus: str, focus_label: str) -> Dict[str, Any]:
                query = " ".join([*fallback_terms[:4], extra_focus]).strip() or question.strip()
                return {"query": query, "focus": focus_label, "channels": ["vector", "keyword"]}

            fillers: List[Dict[str, Any]] = []
            if question_type == "method_flow":
                fillers = [
                    build_extra_query("method framework algorithm", "method"),
                    build_extra_query("training inference architecture", "implementation"),
                ]
            elif question_type == "experiment_setup":
                fillers = [
                    build_extra_query("experiment dataset baseline", "setup"),
                    build_extra_query("evaluation metric implementation", "evaluation"),
                ]
            elif question_type == "results_analysis":
                fillers = [
                    build_extra_query("results performance comparison", "results"),
                    build_extra_query("ablation analysis effect", "analysis"),
                ]
            else:
                fillers = [
                    build_extra_query("paper summary overview", "overview"),
                    build_extra_query("section evidence key findings", "evidence"),
                ]

            for item in fillers:
                if len(rewrite_queries) >= 3 or len(rewrite_queries) >= max_queries:
                    break
                if item["query"] not in {str(entry.get("query", "")) for entry in rewrite_queries if isinstance(entry, dict)}:
                    rewrite_queries.append(item)

        return {
            "question_type": str(data.get("question_type", "other")).strip() or "other",
            "intent_summary": str(data.get("intent_summary", "")).strip(),
            "paper_terms": paper_terms,
            "preferred_sections": preferred_sections,
            "rewrite_queries": rewrite_queries,
        }

    def _extract_fallback_terms(self, text: str, limit: int = 8) -> List[str]:
        tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9_\-]{1,}|[\u4e00-\u9fff]{2,}", text or "")
        seen: List[str] = []
        for token in tokens:
            token = token.strip()
            if token and token not in seen:
                seen.append(token)
        return seen[:limit]

    def _build_qwen_prompt(self, query: str, context: str) -> str:
        return (
            "You are a strict academic QA assistant. Answer only from the provided context.\n"
            "If the context is insufficient, say you cannot determine it.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )

    def stream_qwen_responses(
        self,
        query: str,
        context: str,
        api_key: Optional[str] = None,
        model_name: str = QWEN_MODEL_NAME,
    ) -> Iterator[Dict[str, Any]]:
        """Stream a Qwen Responses API answer chunk by chunk."""
        try:
            if not api_key:
                api_key = QWEN_API_KEY
            if not api_key:
                raise ValueError("Qwen API key not provided")

            client = OpenAI(api_key=api_key, base_url=QWEN_BASE_URL)
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
        show_reasoning: bool = True,
    ) -> str:
        try:
            if not api_key:
                api_key = GENERATION_CONFIG["deepseek_api_key"]
            if not api_key:
                raise ValueError("DeepSeek API key not provided")

            client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer the question."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"},
            ]
            response = client.chat.completions.create(
                model=self.models["deepseek"][model_name],
                messages=messages,
                max_tokens=512,
                stream=False,
            )

            if model_name == "deepseek-r1":
                message = response.choices[0].message
                reasoning = getattr(message, "reasoning_content", None)
                answer = message.content
                if show_reasoning and reasoning:
                    return f"[Reasoning]\n{reasoning}\n\n[Answer]\n{answer}"
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
        鐢熸垚鍥炵瓟骞朵繚瀛樼粨鏋?
        
        鍙傛暟:
            provider: 妯″瀷鎻愪緵鍟嗭紝鍙€夊€间负"openai"銆?qwen"銆?deepseek"
            model_name: 妯″瀷鍚嶇О
            query: 鐢ㄦ埛鏌ヨ
            search_results: 鎼滅储缁撴灉鍒楄〃锛岀敤浜庢瀯寤轰笂涓嬫枃
            api_key: API瀵嗛挜锛堝浜嶢PI璋冪敤锛?
            show_reasoning: 鏄惁鏄剧ず鎺ㄧ悊杩囩▼锛堜粎瀵笵eepSeek鎺ㄧ悊妯″瀷鏈夋晥锛?
            
        杩斿洖:
            鍖呭惈鐢熸垚鍥炵瓟鍜屼繚瀛樿矾寰勭殑瀛楀吀
        """
        try:
            # 鍑嗗涓婁笅鏂?
            context = "\n\n".join([
                f"[Source {i+1}]: {result['text']}"
                for i, result in enumerate(search_results)
            ])
            
            # 鏍规嵁涓嶅悓鎻愪緵鍟嗙敓鎴愬洖绛?
            if provider == "openai":
                response = self._generate_with_openai(model_name, query, context, api_key)
            elif provider == "qwen":
                response = self._generate_with_qwen_responses(query, context, api_key, model_name)
            elif provider == "deepseek":
                response = self._generate_with_deepseek(model_name, query, context, api_key, show_reasoning)
            elif provider == "huggingface":
                raise ValueError("Local HuggingFace generation has been disabled; use openai, qwen, or deepseek.")
            else:
                raise ValueError(f"Unsupported provider: {provider}")
                
            # 鍑嗗淇濆瓨鐨勭粨鏋?
            result = {
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "provider": provider,
                "model": model_name,
                "response": response,
                "context": search_results
            }
            
            # 鐢熸垚鏂囦欢鍚嶅苟淇濆瓨
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
        鑾峰彇鍙敤鐨勬ā鍨嬪垪琛?
        
        杩斿洖:
            鍖呭惈鎵€鏈夋敮鎸佹ā鍨嬬殑瀛楀吀
        """
        return self.models 

