import os
import json
import re
import base64
import mimetypes
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
# 启用 MPS 失败时自动回退到 CPU，避免 Apple Silicon 环境下推理直接报错。
# 当前 PyTorch 对 MPS 的支持仍有边界场景，回退是更稳妥的默认行为。
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

logger = logging.getLogger(__name__)

# 阿里云百炼 / 通义千问 API Key。
# 如果你有自己的真实 Key，请通过环境变量注入，不要直接写死在代码里。
QWEN_API_KEY = GENERATION_CONFIG["qwen_api_key"]
QWEN_BASE_URL = GENERATION_CONFIG["qwen_base_url"]
QWEN_MODEL_NAME = GENERATION_CONFIG["qwen_model_name"]
RERANK_QWEN_MODEL_NAME = "qwen3.6-flash"
QWEN_ENABLE_THINKING = False

class GenerationService:
    """
    生成服务类，负责调用不同模型提供商生成结果。
    支持本地模型和 API 调用，并将生成结果保存到文件。
    """
    def __init__(self):
        """
        初始化生成服务，配置支持的模型列表和输出目录。
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
        
        # 确保输出目录存在
        os.makedirs("05-generation-results", exist_ok=True)

    def compress_chunk_for_rerank(
        self,
        chunk_text: str,
        chunk_metadata: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        model_name: str = RERANK_QWEN_MODEL_NAME,
    ) -> str:
        chunk_metadata = chunk_metadata or {}
        chunk_type = str(chunk_metadata.get("chunk_type", "text") or "text").strip().lower()
        if chunk_type in {"figure", "table"}:
            return self._build_asset_rerank_text(chunk_metadata)
        normalized_text = re.sub(r"\s+", " ", str(chunk_text or "")).strip()
        if not normalized_text:
            return self._build_low_information_rerank_text("empty text", "unknown topic")

        prompt = self._build_rerank_chunk_compression_prompt(normalized_text, chunk_metadata)
        structured_fallback = self._build_structured_rerank_text(normalized_text, chunk_metadata)
        try:
            compressed = self.complete_with_qwen(
                prompt,
                api_key=api_key,
                model_name=model_name,
                enable_thinking=False,
            )
            compressed = self._normalize_rerank_chunk_text(compressed)
            if compressed:
                if self._rerank_text_has_evidence_structure(compressed):
                    return compressed
                logger.debug(
                    "Rerank model output did not match card structure for chunk_id=%s, using structured fallback",
                    chunk_metadata.get("chunk_id", chunk_metadata.get("chunk_index", "")),
                )
        except Exception as exc:
            logger.warning(
                "Failed to compress chunk %s for rerank with Qwen, falling back to structured heuristic rerank text: %s",
                chunk_metadata.get("chunk_id", chunk_metadata.get("chunk_index", "")),
                exc,
            )

        return structured_fallback

    def compress_chunks_for_rerank(
        self,
        chunks: List[Dict[str, Any]],
        api_key: Optional[str] = None,
        model_name: str = RERANK_QWEN_MODEL_NAME,
    ) -> List[Dict[str, Any]]:
        compressed_chunks: List[Dict[str, Any]] = []
        logger.info(
            "Preparing rerank_text for %d chunks with model=%s enable_thinking=%s",
            len(chunks),
            model_name,
            False,
        )

        for chunk in chunks:
            updated_chunk = dict(chunk)
            metadata = dict(updated_chunk.get("metadata", {}) or {})
            chunk_id = metadata.get("chunk_id", metadata.get("chunk_index", len(compressed_chunks) + 1))
            raw_content = str(updated_chunk.get("content", "") or "")
            rerank_text = self.compress_chunk_for_rerank(
                chunk_text=raw_content,
                chunk_metadata=metadata,
                api_key=api_key,
                model_name=model_name,
            )

            metadata["rerank_text"] = rerank_text
            metadata["rerank_text_model"] = model_name
            metadata["rerank_text_generated_at"] = datetime.now().isoformat()
            updated_chunk["metadata"] = metadata
            updated_chunk["rerank_text"] = rerank_text

            logger.info(
                "Prepared rerank_text for chunk_id=%s model=%s enable_thinking=%s raw_preview=%s rerank_preview=%s",
                chunk_id,
                model_name,
                False,
                self._preview_text(raw_content, 120),
                self._preview_text(rerank_text, 120),
            )
            compressed_chunks.append(updated_chunk)

        return compressed_chunks

    def _build_asset_rerank_text(self, chunk_metadata: Dict[str, Any]) -> str:
        chunk_type = str(chunk_metadata.get("chunk_type", "text") or "text").strip().lower()
        summary = str(chunk_metadata.get("asset_summary", "") or "").strip()
        preview = str(chunk_metadata.get("asset_preview_text", "") or "").strip()
        section_title = str(chunk_metadata.get("section_title", "") or "").strip()
        section_path = str(chunk_metadata.get("section_path", "") or "").strip()
        page_number = str(chunk_metadata.get("page_number", "") or chunk_metadata.get("page_range", "") or "").strip()
        label = "Figure evidence" if chunk_type == "figure" else "Table evidence"
        parts = [
            label,
            summary,
            preview if preview and preview != summary else "",
            section_title,
            section_path,
            f"page {page_number}".strip() if page_number else "",
        ]
        return "\n".join(part for part in parts if part).strip()
        
    def _load_huggingface_model(self, model_name: str):
        """
        加载 HuggingFace 模型。
        
        参数:
            model_name: 模型名称，对应 self.models["huggingface"] 中的键。
            
        返回:
            model: 加载后的模型。
            tokenizer: 对应的分词器。
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
        使用 HuggingFace 模型生成回答。
        
        参数:
            model_name: 模型名称。
            query: 用户查询。
            context: 上下文信息。
            max_length: 生成文本的最大长度。
            
        返回:
            生成的回答文本。
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
        使用 OpenAI API 生成回答。
        
        参数:
            model_name: 模型名称。
            query: 用户查询。
            context: 上下文信息。
            api_key: OpenAI API 密钥；如果不提供则从环境变量读取。
            
        返回:
            生成的回答文本。
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
        enable_thinking: bool = QWEN_ENABLE_THINKING,
        image_inputs: Optional[List[Dict[str, Any]]] = None,
        asset_metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        使用阿里云百炼兼容的 OpenAI Responses API 生成答案。

        这里采用 Qwen3.6-Plus，输入由检索到的上下文和问题组成。
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

            response = client.responses.create(
                model=model_name,
                input=self._build_qwen_input(
                    query=query,
                    context=context,
                    image_inputs=image_inputs,
                    asset_metadata=asset_metadata,
                ),
                extra_body={"enable_thinking": enable_thinking},
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
        enable_thinking: bool = QWEN_ENABLE_THINKING,
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
                extra_body={"enable_thinking": enable_thinking},
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

    def rewrite_query_for_rerank(
        self,
        question: str,
        paper_context: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        model_name: str = QWEN_MODEL_NAME,
    ) -> str:
        return self.build_rerank_query(
            question,
            api_key=api_key,
            model_name=model_name,
        )

    def build_rerank_query(
        self,
        original_question: str,
        api_key: Optional[str] = None,
        model_name: str = QWEN_MODEL_NAME,
    ) -> str:
        normalized_question = re.sub(r"\s+", " ", (original_question or "")).strip()
        rerank_query = self._build_evidence_selection_rerank_query(normalized_question)
        logger.debug("original_question=%s", normalized_question)
        logger.debug("actual_rerank_query=%s", rerank_query)
        return rerank_query

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

    def _fallback_rerank_query(self, question: str) -> str:
        return self._build_evidence_selection_rerank_query(question)

    def _build_evidence_selection_rerank_query(self, original_question: str) -> str:
        normalized_question = re.sub(r"\s+", " ", (original_question or "")).strip()
        base_query = (
            "Select the passage that most directly supports an answer to the user's question. "
            "Prefer evidence-bearing chunks with explicit facts, definitions, steps, causes, results, comparisons, or other answerable statements; "
            "down-rank passages that are only loosely topic-related or background. "
        )
        if normalized_question:
            return f"{base_query}Original question: {normalized_question}"
        return base_query.rstrip()

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

    def _build_rerank_chunk_compression_prompt(
        self,
        chunk_text: str,
        chunk_metadata: Dict[str, Any],
    ) -> str:
        section_title = str(chunk_metadata.get("section_title", "") or "").strip()
        section_path = str(chunk_metadata.get("section_path", "") or "").strip()
        page_range = str(chunk_metadata.get("page_range", "") or "").strip()
        word_count = int(chunk_metadata.get("word_count", len(chunk_text.split())) or 0)
        role_label, useful_scope = self._infer_rerank_card_fields(chunk_text, chunk_metadata)

        return (
            "You are preparing a passage for a reranker in an academic RAG system.\n"
            "Transform the chunk into a compact evidence card for reranking.\n"
            "Requirements:\n"
            "1. Use exactly this structure:\n"
            "Role: {content_role}\n"
            "Useful for: {positive_question_scope}\n"
            "Core evidence:\n"
            "{concise_core_content}\n"
            "2. Keep Role short and specific.\n"
            "3. Useful for must only describe positive question scope.\n"
            "4. Core evidence must be 1 to 3 short sentences.\n"
            "5. Do not add negative scope labels or generic summary padding.\n"
            "6. If the chunk is low-information, use the low-information template and do not elaborate.\n"
            "7. Do not invent facts that are not in the chunk.\n"
            "8. Output plain text only.\n\n"
            f"Section title: {section_title or 'N/A'}\n"
            f"Section path: {section_path or 'N/A'}\n"
            f"Page range: {page_range or 'N/A'}\n"
            f"Word count: {word_count}\n\n"
            f"Likely role: {role_label}\n"
            f"Likely useful scope: {useful_scope}\n\n"
            "Chunk:\n"
            f"{chunk_text}\n\n"
            "Evidence card:"
        )

    def _normalize_rerank_chunk_text(self, text: str) -> str:
        normalized = re.sub(r"\s+", " ", str(text or "")).strip()
        normalized = re.sub(r"^(compressed rerank text|evidence-value rerank text|evidence card|rerank text|answer)\s*:\s*", "", normalized, flags=re.IGNORECASE)
        return normalized.strip()

    def _build_structured_rerank_text(self, chunk_text: str, chunk_metadata: Dict[str, Any]) -> str:
        normalized = re.sub(r"\s+", " ", str(chunk_text or "")).strip()
        if self._looks_like_low_information_chunk(normalized, chunk_metadata):
            return self._build_low_information_rerank_text()

        role_label, useful_scope = self._infer_rerank_card_fields(normalized, chunk_metadata)
        core_content = self._extract_rerank_core_content(normalized, chunk_metadata, role_label)
        return self._compose_rerank_card_text(role_label, useful_scope, core_content)

    def _compose_rerank_card_text(self, role_label: str, useful_scope: str, core_content: str) -> str:
        return (
            f"Role: {role_label}.\n"
            f"Useful for: {useful_scope}.\n"
            "Core evidence:\n"
            f"{core_content}"
        )

    def _build_low_information_rerank_text(self) -> str:
        return (
            "Role: low-information.\n"
            "Useful for: document metadata or structure identification.\n"
            "Core evidence:\n"
            "Not standalone evidence."
        )

    def _infer_rerank_card_fields(self, text: str, chunk_metadata: Dict[str, Any]) -> tuple[str, str]:
        normalized = re.sub(r"\s+", " ", str(text or "")).strip()
        lowered = normalized.lower()
        section_title = str(chunk_metadata.get("section_title", "") or "").strip().lower()
        section_path = str(chunk_metadata.get("section_path", "") or "").strip().lower()
        section_tags = {str(tag).strip().lower() for tag in (chunk_metadata.get("section_tags", []) or []) if str(tag).strip()}
        heading_hint = f"{section_path} {section_title}".strip()

        if self._looks_like_low_information_chunk(normalized, chunk_metadata):
            return (
                "low-information",
                "document metadata or structure identification",
            )

        if any(tag in {"method", "methods", "model", "approach", "framework"} for tag in section_tags) or re.search(
            r"\b(method|methods|approach|framework|architecture|pipeline|algorithm|training loop|reward|optimization|objective)\b",
            lowered,
        ):
            if re.search(r"\breward\b|\bpolicy\b|\bgrpo\b|\bppo\b|\brl\b|\breinforcement learning\b", lowered):
                return (
                    "method / training objective",
                    "method, training, reward design, and optimization process questions",
                )
            if re.search(r"\balgorithm\b|\bpseudocode\b|\bstep\b|\bprocedure\b", lowered):
                return (
                    "algorithm / procedure",
                    "algorithm flow, step-by-step procedure, and component interaction questions",
                )
            return (
                "method / framework",
                "method, framework, component, or pipeline questions",
            )

        if any(tag in {"experiment", "experiments", "results", "result", "analysis", "ablation"} for tag in section_tags) or re.search(
            r"\b(experiment|experiments|results?|analysis|ablation|baseline|benchmarks?|table|figure|score|metric|accuracy|f1|recall|precision|improvement|dataset|setting|hyperparameter)\b",
            lowered,
        ):
            if re.search(r"\bablation\b|\bremove\b|\bwithout\b|\bvariant\b|\bcomponent\b", lowered):
                return (
                    "ablation",
                    "ablation, component contribution, and design choice comparison questions",
                )
            if re.search(r"\bsetup\b|\bimplementation\b|\bhyperparameter\b|\btraining details\b|\bdata\b|\bdataset\b|\bbaseline\b", lowered):
                return (
                    "experiment setup",
                    "dataset, baseline, metric, implementation, and evaluation setting questions",
                )
            return (
                "result / analysis",
                "result interpretation, performance comparison, and empirical finding questions",
            )

        if any(tag in {"appendix", "supplementary"} for tag in section_tags) or re.search(
            r"\b(appendix|supplementary|additional results|extra analysis|case study|qualitative|transferability)\b",
            lowered,
        ):
            return (
                "appendix / supplementary",
                "supplementary analysis, transferability, or additional evidence questions",
            )

        if any(tag in {"limitation", "limitations", "ethics", "ethic", "broader-impact"} for tag in section_tags) or re.search(
            r"\b(limitation|limitations|ethic|ethics|bias|risk|responsible|misuse|privacy|safety)\b",
            lowered,
        ):
            return (
                "limitation / ethics",
                "limitations, bias, safety, or responsible-use questions",
            )

        if any(tag in {"related-work", "background", "related"} for tag in section_tags) or re.search(
            r"\b(related work|background|prior work|motivation|survey)\b",
            lowered,
        ):
            return (
                "background / related work",
                "motivation, prior work, and literature context questions",
            )

        if re.search(r"\bconclusion\b|\bsummary\b", section_title + " " + section_path + " " + lowered):
            return (
                "conclusion",
                "takeaway, closing claim, or paper summary questions",
            )

        return (
            "substantive evidence",
            "local factual claims, definitions, or conceptual explanation questions",
        )

    def _extract_rerank_core_content(self, text: str, chunk_metadata: Dict[str, Any], role_label: str) -> str:
        normalized = re.sub(r"\s+", " ", str(text or "")).strip()
        section_title = str(chunk_metadata.get("section_title", "") or "").strip()

        if not normalized:
            return "No substantive text was provided."

        if section_title and section_title.lower() not in normalized.lower():
            normalized = f"{section_title}. {normalized}"

        if len(normalized) <= 360:
            return normalized

        sentences = re.split(r"(?<=[.!?。！？；;])\s+", normalized)
        selected: List[str] = []
        key_terms = self._role_key_terms(role_label)
        for sentence in sentences:
            cleaned = sentence.strip()
            if not cleaned:
                continue
            lowered = cleaned.lower()
            if key_terms and any(term in lowered for term in key_terms):
                selected.append(cleaned)
            elif not selected and len(cleaned) > 32:
                selected.append(cleaned)
            if len(selected) >= 3:
                break

        if not selected:
            selected = [self._preview_text(normalized, 320)]
        return " ".join(selected).strip()

    def _role_key_terms(self, role_label: str) -> List[str]:
        role = role_label.lower()
        if "method" in role or "algorithm" in role or "framework" in role or "training" in role:
            return ["method", "framework", "algorithm", "training", "reward", "policy", "optimization"]
        if "experiment" in role or "result" in role or "ablation" in role:
            return ["experiment", "result", "baseline", "ablation", "table", "dataset", "metric", "score"]
        if "appendix" in role or "supplementary" in role:
            return ["appendix", "supplementary", "additional", "case study", "analysis"]
        if "ethics" in role or "limitation" in role:
            return ["limit", "ethic", "bias", "risk", "safety", "privacy"]
        return []

    def _rerank_text_has_evidence_structure(self, text: str) -> bool:
        lowered = str(text or "").lower()
        return "role:" in lowered and "useful for:" in lowered and "core evidence:" in lowered

    def _looks_like_low_information_chunk(self, text: str, chunk_metadata: Dict[str, Any]) -> bool:
        normalized = str(text or "").strip()
        if not normalized:
            return True

        word_count = len(normalized.split())
        lowered = normalized.lower()
        section_title = str(chunk_metadata.get("section_title", "") or "").strip().lower()

        if word_count <= 14 and "\n" not in normalized and normalized == normalized.title():
            return True
        if any(token in lowered for token in ("references", "bibliography")):
            return True
        if re.match(r"^\[?\d+\]?\s+[A-Z]", normalized):
            return True
        if re.search(r"\b(email|university|institute|department)\b", lowered) and word_count <= 40:
            return True
        if re.search(r"\barxiv\b|\bdoi\b|http[s]?://", lowered) and word_count <= 40:
            return True
        if section_title and normalized.lower() == section_title:
            return True
        if word_count <= 8:
            return True
        return False

    def _preview_text(self, text: str, limit: int = 120) -> str:
        normalized = re.sub(r"\s+", " ", str(text or "")).strip()
        if len(normalized) <= limit:
            return normalized
        return normalized[:limit].rstrip()

    def _build_qwen_prompt(self, query: str, context: str) -> str:
        return (
            "You are a strict academic QA assistant. Answer only from the provided context.\n"
            "If the context is insufficient, say you cannot determine it.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )

    def _build_qwen_input(
        self,
        query: str,
        context: str,
        image_inputs: Optional[List[Dict[str, Any]]] = None,
        asset_metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> Any:
        image_inputs = image_inputs or []
        asset_metadata = asset_metadata or []
        if not image_inputs:
            return self._build_qwen_prompt(query, context)

        evidence_lines = []
        for index, item in enumerate(asset_metadata, start=1):
            evidence_lines.append(
                f"[Image {index}] page={item.get('page_number', '')} summary={item.get('asset_summary', '')} section={item.get('section_path', '')}"
            )

        intro_text = (
            "You are a strict academic QA assistant. Answer only from the provided context and image evidence.\n"
            "If the context is insufficient, say you cannot determine it.\n\n"
            f"Text Context:\n{context}\n\n"
            f"Image Evidence Notes:\n{chr(10).join(evidence_lines) if evidence_lines else 'None'}\n\n"
            f"Question: {query}"
        )
        content = [{"type": "input_text", "text": intro_text}]
        for image in image_inputs:
            image_path = str(image.get("image_path", "") or "").strip()
            if not image_path:
                continue
            content.append(
                {
                    "type": "input_image",
                    "image_url": self._image_path_to_data_url(image_path),
                }
            )
        return [{"role": "user", "content": content}]

    def stream_qwen_responses(
        self,
        query: str,
        context: str,
        api_key: Optional[str] = None,
        model_name: str = QWEN_MODEL_NAME,
        enable_thinking: bool = QWEN_ENABLE_THINKING,
        image_inputs: Optional[List[Dict[str, Any]]] = None,
        asset_metadata: Optional[List[Dict[str, Any]]] = None,
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
                input=self._build_qwen_input(
                    query=query,
                    context=context,
                    image_inputs=image_inputs,
                    asset_metadata=asset_metadata,
                ),
                stream=True,
                extra_body={"enable_thinking": enable_thinking},
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
        show_reasoning: bool = True,
        image_inputs: Optional[List[Dict[str, Any]]] = None,
        asset_metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict:
        """
        生成回答并保存结果。
        
        参数:
            provider: 模型提供商，可选值为 "openai"、"qwen"、"deepseek"。
            model_name: 模型名称。
            query: 用户查询。
            search_results: 检索结果列表，用于构建上下文。
            api_key: API 密钥，对应不同提供商。
            show_reasoning: 是否显示推理过程（仅对 DeepSeek 推理模型有效）。
            
        返回:
            包含生成回答和保存路径的字典。
        """
        try:
            # 准备上下文
            context = "\n\n".join([
                f"[Source {i+1}]: {result['text']}"
                for i, result in enumerate(search_results)
            ])
            
            # 根据不同提供商生成回答
            if provider == "openai":
                response = self._generate_with_openai(model_name, query, context, api_key)
            elif provider == "qwen":
                response = self._generate_with_qwen_responses(
                    query,
                    context,
                    api_key,
                    model_name,
                    enable_thinking=QWEN_ENABLE_THINKING,
                    image_inputs=image_inputs,
                    asset_metadata=asset_metadata,
                )
            elif provider == "deepseek":
                response = self._generate_with_deepseek(model_name, query, context, api_key, show_reasoning)
            elif provider == "huggingface":
                raise ValueError("Local HuggingFace generation has been disabled; use openai, qwen, or deepseek.")
            else:
                raise ValueError(f"Unsupported provider: {provider}")
                
            # 准备保存结果
            result = {
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "provider": provider,
                "model": model_name,
                "response": response,
                "context": search_results,
                "image_inputs": image_inputs or [],
                "asset_metadata": asset_metadata or [],
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
        获取可用模型列表。
        
        返回:
            包含所有支持模型的字典。
        """
        return self.models 

    def _image_path_to_data_url(self, image_path: str) -> str:
        if not image_path:
            raise ValueError("Image path is required for multimodal generation")
        if not os.path.exists(image_path):
            raise ValueError(f"Image path does not exist: {image_path}")
        mime_type = mimetypes.guess_type(image_path)[0] or "image/png"
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

