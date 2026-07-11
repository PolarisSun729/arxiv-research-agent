import os
import json
import re
import base64
import mimetypes
import io
from datetime import datetime
from typing import List, Dict, Optional, Iterator, Any, Tuple
import logging
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from openai import OpenAI
import requests
from utils.model_utils import get_huggingface_model_path
from utils.config import GENERATION_CONFIG
from utils.storage_paths import resolve_backend_artifact_path
# 启用 MPS 失败时自动回退到 CPU，避免 Apple Silicon 环境下推理直接报错。
# 当前 PyTorch 对 MPS 的支持仍有边界场景，回退是更稳妥的默认行为。
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

logger = logging.getLogger(__name__)

# 阿里云百炼 / 通义千问 API Key。
# 如果你有自己的真实 Key，请通过环境变量注入，不要直接写死在代码里。
QWEN_API_KEY = GENERATION_CONFIG["qwen_api_key"]
QWEN_BASE_URL = GENERATION_CONFIG["qwen_base_url"]
QWEN_SMALL_MODEL_NAME = GENERATION_CONFIG["small_qwen_model_name"]
QWEN_LARGE_MODEL_NAME = GENERATION_CONFIG["large_qwen_model_name"]
QWEN_RERANK_COMPRESS_MODEL_NAME = GENERATION_CONFIG["qwen_rerank_compress_model_name"]
QWEN_RERANK_COMPRESS_ENABLE_THINKING = GENERATION_CONFIG["qwen_rerank_compress_enable_thinking"]
HF_GENERATE_MAX_LENGTH = GENERATION_CONFIG["huggingface_generate_max_length"]
HF_GENERATE_TEMPERATURE = GENERATION_CONFIG["huggingface_generate_temperature"]
HF_GENERATE_DO_SAMPLE = GENERATION_CONFIG["huggingface_generate_do_sample"]
REWRITE_QUERY_MAX_QUERIES_DEFAULT = GENERATION_CONFIG["rewrite_query_max_queries_default"]
PLAN_QUERY_MAX_QUERIES_DEFAULT = GENERATION_CONFIG["plan_query_max_queries_default"]
QWEN_TASK_MODEL_ROLES = dict(GENERATION_CONFIG.get("task_model_roles", {}))

# DashScope Responses API 会在入站阶段按请求体字节数拒绝超大请求；这里用软限制预留
# JSON、模型名和 extra_body 的余量，避免图片 base64 后把最终 QA 请求撑爆。
QWEN_RESPONSES_REQUEST_BODY_LIMIT_BYTES = int(
    GENERATION_CONFIG.get("qwen_responses_request_body_limit_bytes") or 6_291_456
)
QWEN_RESPONSES_REQUEST_BODY_SOFT_LIMIT_BYTES = int(
    GENERATION_CONFIG.get("qwen_responses_request_body_soft_limit_bytes") or 5_500_000
)
QWEN_RESPONSES_IMAGE_TARGET_BYTES = int(
    GENERATION_CONFIG.get("qwen_responses_image_target_bytes") or 1_500_000
)
QWEN_RESPONSES_MAX_IMAGE_COUNT = int(
    GENERATION_CONFIG.get("qwen_responses_max_image_count") or 2
)
QWEN_RESPONSES_IMAGE_COMPRESSION_STEPS = (
    (1600, 85),
    (1280, 75),
    (1024, 65),
)

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
                "qwen3.6-plus": QWEN_LARGE_MODEL_NAME,
                "qwen3.6-flash": QWEN_SMALL_MODEL_NAME,
                "large": QWEN_LARGE_MODEL_NAME,
                "small": QWEN_SMALL_MODEL_NAME,
            },
            "deepseek": {
                "deepseek-v3": "deepseek-chat",
                "deepseek-r1": "deepseek-reasoner",
            }
        }
        
        # 生成结果是后端运行产物；路径固定到 backend 下，避免工作目录不同导致写入根目录。
        self.generation_results_dir = resolve_backend_artifact_path(
            "05-generation-results",
            option_name="GENERATION_RESULTS_DIR",
        )
        os.makedirs(self.generation_results_dir, exist_ok=True)

    def _normalize_task_type(self, task_type: Optional[str]) -> str:
        # 任务类型只作为路由提示使用，统一做一次清洗，避免空字符串污染日志和配置查询。
        return str(task_type or "").strip().lower()

    def _resolve_qwen_model_selection(
        self,
        *,
        task_type: Optional[str] = None,
        model_name: Optional[str] = None,
        model_role: Optional[str] = None,
        default_role: str = "large",
    ) -> Dict[str, str]:
        """
        根据任务类型选择 Qwen 模型。

        这里的职责是把“任务类型 -> 小/大模型 -> 具体模型名”的决策收口，
        业务调用方只需要传 task_type，不再自己硬编码模型名。
        """
        normalized_task_type = self._normalize_task_type(task_type)
        requested_role = str(model_role or "").strip().lower()
        configured_role = str(QWEN_TASK_MODEL_ROLES.get(normalized_task_type, "") or "").strip().lower()
        selected_role = requested_role or configured_role or default_role
        selected_model = str(model_name or "").strip()
        routing_source = "explicit_model_name" if selected_model else "task_route"
        fallback_reason = ""

        if not selected_model:
            if selected_role == "small":
                selected_model = QWEN_SMALL_MODEL_NAME
            else:
                selected_model = QWEN_LARGE_MODEL_NAME

            if not configured_role and normalized_task_type and normalized_task_type not in {"default", "general_generation"}:
                fallback_reason = f"task_type={normalized_task_type} 采用默认模型角色={selected_role}"
                routing_source = "default"
        else:
            if selected_model == QWEN_SMALL_MODEL_NAME:
                selected_role = "small"
            elif selected_model == QWEN_LARGE_MODEL_NAME:
                selected_role = "large"
            else:
                selected_role = requested_role or "custom"
                routing_source = "explicit_override"

        logger.debug(
            "Qwen模型路由 task_type=%s model_role=%s selected_model=%s routing_source=%s fallback_reason=%s",
            normalized_task_type or "default",
            selected_role,
            selected_model,
            routing_source,
            fallback_reason or "",
        )
        return {
            "task_type": normalized_task_type or "default",
            "model_role": selected_role,
            "selected_model": selected_model,
            "routing_source": routing_source,
            "fallback_reason": fallback_reason,
        }

    def compress_chunk_for_rerank(
        self,
        chunk_text: str,
        chunk_metadata: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        model_name: str = QWEN_RERANK_COMPRESS_MODEL_NAME,
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
        model_name: str = QWEN_RERANK_COMPRESS_MODEL_NAME,
    ) -> List[Dict[str, Any]]:
        compressed_chunks: List[Dict[str, Any]] = []
        logger.debug(
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

            logger.debug(
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
        max_length: int = HF_GENERATE_MAX_LENGTH
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
                temperature=HF_GENERATE_TEMPERATURE,
                do_sample=HF_GENERATE_DO_SAMPLE
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
                temperature=GENERATION_CONFIG["openai_chat_temperature"],
                max_tokens=GENERATION_CONFIG["openai_chat_max_tokens"]
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
        model_name: Optional[str] = None,
        task_type: Optional[str] = None,
        enable_thinking: bool = QWEN_RERANK_COMPRESS_ENABLE_THINKING,
        image_inputs: Optional[List[Dict[str, Any]]] = None,
        asset_metadata: Optional[List[Dict[str, Any]]] = None,
        request_debug: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        使用阿里云百炼兼容的 OpenAI Responses API 生成答案。

        这里会根据 task_type 选择小模型或大模型；如果外部显式传入 model_name，则优先尊重显式配置。
        """
        try:
            if not api_key:
                api_key = QWEN_API_KEY
            if not api_key:
                raise ValueError("Qwen API key not provided")

            model_selection = self._resolve_qwen_model_selection(
                task_type=task_type,
                model_name=model_name,
                default_role="large",
            )
            model_name = model_selection["selected_model"]

            client = OpenAI(
                api_key=api_key,
                base_url=QWEN_BASE_URL,
            )

            qwen_input, input_debug = self._build_qwen_input_with_debug(
                query=query,
                context=context,
                image_inputs=image_inputs,
                asset_metadata=asset_metadata,
            )
            if request_debug is not None:
                request_debug.update(input_debug)

            response = client.responses.create(
                model=model_name,
                input=qwen_input,
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
        model_name: Optional[str] = None,
        task_type: Optional[str] = None,
        enable_thinking: bool = QWEN_RERANK_COMPRESS_ENABLE_THINKING,
    ) -> str:
        try:
            if not api_key:
                api_key = QWEN_API_KEY
            if not api_key:
                raise ValueError("Qwen API key not provided")

            model_selection = self._resolve_qwen_model_selection(
                task_type=task_type,
                model_name=model_name,
                default_role="large",
            )
            model_name = model_selection["selected_model"]

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
        max_queries: int = REWRITE_QUERY_MAX_QUERIES_DEFAULT,
        paper_context: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
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
        model_name: Optional[str] = None,
        intent_profile: Optional[Dict[str, Any]] = None,
    ) -> str:
        return self.build_rerank_query(
            question,
            api_key=api_key,
            model_name=model_name,
            intent_profile=intent_profile,
        )

    def build_rerank_query(
        self,
        original_question: str,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        intent_profile: Optional[Dict[str, Any]] = None,
    ) -> str:
        normalized_question = re.sub(r"\s+", " ", (original_question or "")).strip()
        rerank_query = self._build_evidence_selection_rerank_query(normalized_question, intent_profile=intent_profile)
        logger.debug("original_question=%s", normalized_question)
        if intent_profile:
            logger.debug(
                "intent_profile_for_rerank main_intent=%s sub_intents=%s confidence=%s",
                intent_profile.get("main_intent"),
                intent_profile.get("sub_intents"),
                intent_profile.get("confidence"),
            )
        logger.debug("actual_rerank_query=%s", rerank_query)
        return rerank_query

    def plan_queries_for_retrieval(
        self,
        question: str,
        max_queries: int = PLAN_QUERY_MAX_QUERIES_DEFAULT,
        paper_context: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        intent_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        paper_context = paper_context or {}
        title = str(paper_context.get("title", "") or "").strip()
        abstract = str(paper_context.get("abstract", "") or "").strip()
        section_titles = [str(item).strip() for item in (paper_context.get("section_titles", []) or []) if str(item).strip()]
        candidate_terms = [str(item).strip() for item in (paper_context.get("candidate_terms", []) or []) if str(item).strip()]
        main_intent = str((intent_profile or {}).get("main_intent", "other") or "other").strip() or "other"
        intent_summary = str((intent_profile or {}).get("intent_summary", "") or "").strip()
        preferred_sections = [str(item).strip() for item in (intent_profile or {}).get("preferred_sections", []) or [] if str(item).strip()]
        sub_intents = [str(item).strip() for item in (intent_profile or {}).get("sub_intents", []) or [] if str(item).strip()]

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
            f"10. Current intent profile: main_intent={main_intent}, intent_summary={intent_summary or 'N/A'}, sub_intents={', '.join(sub_intents) or 'N/A'}, preferred_sections={', '.join(preferred_sections) or 'N/A'}.\n\n"
            f"User question: {question}\n\n"
            f"Paper title: {title or 'N/A'}\n"
            f"Paper abstract: {abstract[:1800] or 'N/A'}\n"
            f"Section titles: {', '.join(section_titles[:16]) or 'N/A'}\n"
            f"Candidate paper terms: {', '.join(candidate_terms[:24]) or 'N/A'}"
        )
        response = self.complete_with_qwen(
            prompt,
            api_key=api_key,
            model_name=model_name,
            task_type="query_planning",
        )
        data = json.loads(self._extract_json_block(response))
        return self._normalize_query_plan(data, question, max_queries=max_queries, paper_context=paper_context)

    def generate_hyde_document(
        self,
        question: str,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
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
        return self.complete_with_qwen(
            prompt,
            api_key=api_key,
            model_name=model_name,
            task_type="hyde_generation",
        ).strip()

    def _fallback_rerank_query(self, question: str, intent_profile: Optional[Dict[str, Any]] = None) -> str:
        return self._build_evidence_selection_rerank_query(question, intent_profile=intent_profile)

    def _build_evidence_selection_rerank_query(self, original_question: str, intent_profile: Optional[Dict[str, Any]] = None) -> str:
        normalized_question = re.sub(r"\s+", " ", (original_question or "")).strip()
        base_query = (
            "Select the passage that most directly supports an answer to the user's question. "
            "Prefer evidence-bearing chunks with explicit facts, definitions, steps, causes, results, comparisons, or other answerable statements; "
            "down-rank passages that are only loosely topic-related or background. "
        )
        intent_clause = self._build_intent_rerank_clause(intent_profile)
        if normalized_question:
            return f"{base_query}{intent_clause}Original question: {normalized_question}"
        return f"{base_query}{intent_clause}".rstrip()

    def _legacy_intent_bucket(self, intent: str) -> str:
        intent = str(intent or "other").strip().lower() or "other"
        aliases = {
            "contribution": "summary",
            "paper_overview": "summary",
            "method_flow": "method",
            "implementation_detail": "method",
            "definition": "method",
            "experiment_setup": "experiment",
            "result_analysis": "experiment",
            "results_analysis": "experiment",
            "comparison": "comparison",
            "dataset": "dataset",
            "limitation": "limitation",
            "figure_table": "figure_table",
            "other": "other",
            "summary": "summary",
            "method": "method",
            "experiment": "experiment",
        }
        return aliases.get(intent, intent)

    def _build_intent_rerank_clause(self, intent_profile: Optional[Dict[str, Any]]) -> str:
        if not intent_profile:
            return ""

        main_intent = self._legacy_intent_bucket(intent_profile.get("main_intent", "other"))
        preferred_sections = [str(item).strip() for item in (intent_profile.get("preferred_sections", []) or []) if str(item).strip()]
        sub_intents = [str(item).strip() for item in (intent_profile.get("sub_intents", []) or []) if str(item).strip()]

        intent_clauses = {
            "summary": "Prioritize abstract, introduction, and conclusion passages that state the paper's main contribution or findings. ",
            "method": "Prioritize method, architecture, training, inference, and implementation details. ",
            "experiment": "Prioritize experiment, evaluation, results, metric, baseline, and ablation evidence. ",
            "comparison": "Prioritize direct baseline comparisons and ablation evidence. ",
            "dataset": "Prioritize dataset, corpus, benchmark, split, and data description passages. ",
            "limitation": "Prioritize limitations, failure cases, discussion, and future work. ",
            "figure_table": "Prioritize figure captions, table captions, appendix references, and visual explanations. ",
        }
        parts = [intent_clauses.get(main_intent, "")]
        if "paper_overview" in sub_intents:
            parts.append("Prefer passages that summarize the paper at a high level. ")
        if "evidence_seeking" in sub_intents:
            parts.append("Prefer passages that provide direct answer-bearing evidence. ")
        if "result_check" in sub_intents:
            parts.append("Prefer passages with concrete numbers, metrics, and outcome descriptions. ")
        if "table_lookup" in sub_intents:
            parts.append("Prefer passages tied to figures, tables, captions, or appendix visual material. ")
        if "deep_method" in sub_intents:
            parts.append("Prefer passages that explain the technical pipeline and implementation. ")
        if preferred_sections:
            parts.append(f"Favor sections such as: {', '.join(preferred_sections[:4])}. ")
        return "".join(parts)

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
            question_type = self._legacy_intent_bucket(data.get("question_type", "other"))
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
            "question_type": self._legacy_intent_bucket(data.get("question_type", "other")),
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
        qwen_input, _ = self._build_qwen_input_with_debug(
            query=query,
            context=context,
            image_inputs=image_inputs,
            asset_metadata=asset_metadata,
        )
        return qwen_input

    def _build_qwen_input_with_debug(
        self,
        query: str,
        context: str,
        image_inputs: Optional[List[Dict[str, Any]]] = None,
        asset_metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        image_inputs = image_inputs or []
        asset_metadata = asset_metadata or []
        if not image_inputs:
            prompt = self._build_qwen_prompt(query, context)
            return prompt, {
                "qwen_multimodal_enabled": False,
                "candidate_image_count": 0,
                "sent_image_count": 0,
                "dropped_image_count": 0,
                "estimated_request_bytes": len(prompt.encode("utf-8")),
                "images": [],
            }

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
        image_parts, image_debug = self._prepare_qwen_image_content_parts(
            image_inputs=image_inputs,
            base_content=content,
        )
        content.extend(image_parts)
        qwen_input = [{"role": "user", "content": content}]
        image_debug["estimated_request_bytes"] = self._estimate_qwen_input_bytes(qwen_input)
        return qwen_input, image_debug

    def _prepare_qwen_image_content_parts(
        self,
        *,
        image_inputs: List[Dict[str, Any]],
        base_content: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        debug: Dict[str, Any] = {
            "qwen_multimodal_enabled": True,
            "request_body_limit_bytes": QWEN_RESPONSES_REQUEST_BODY_LIMIT_BYTES,
            "request_body_soft_limit_bytes": QWEN_RESPONSES_REQUEST_BODY_SOFT_LIMIT_BYTES,
            "image_target_bytes": QWEN_RESPONSES_IMAGE_TARGET_BYTES,
            "max_image_count": QWEN_RESPONSES_MAX_IMAGE_COUNT,
            "candidate_image_count": len(image_inputs),
            "sent_image_count": 0,
            "dropped_image_count": 0,
            "base_request_bytes": self._estimate_qwen_input_bytes([{"role": "user", "content": base_content}]),
            "estimated_request_bytes": 0,
            "images": [],
        }
        selected_parts: List[Dict[str, Any]] = []

        for image in image_inputs:
            item_debug = self._new_qwen_image_debug_item(image)
            if debug["sent_image_count"] >= QWEN_RESPONSES_MAX_IMAGE_COUNT:
                self._mark_qwen_image_skipped(item_debug, "max_image_count_exceeded")
                debug["images"].append(item_debug)
                continue

            image_path = str(image.get("image_path", "") or "").strip()
            if not image_path:
                self._mark_qwen_image_skipped(item_debug, "missing_image_path")
                debug["images"].append(item_debug)
                continue

            try:
                prepared = self._prepare_qwen_image_data_url(
                    image_path,
                    target_bytes=QWEN_RESPONSES_IMAGE_TARGET_BYTES,
                )
                item_debug.update(prepared.get("debug", {}))
                data_url = str(prepared.get("data_url") or "")
            except Exception as exc:
                # 图片处理失败不应中断 QA；文本证据和图片摘要仍可支撑一次降级回答。
                item_debug["error"] = str(exc)
                logger.warning(
                    "Qwen image input skipped due to preparation error: source_id=%s image_path=%s error=%s",
                    item_debug.get("source_id"),
                    image_path,
                    exc,
                )
                data_url = ""

            if not data_url:
                self._mark_qwen_image_skipped(
                    item_debug,
                    str(item_debug.get("skip_reason") or "image_preparation_failed"),
                )
                debug["images"].append(item_debug)
                continue

            label_part = {
                "type": "input_text",
                "text": self._build_qwen_image_label_text(image),
            }
            image_part = {"type": "input_image", "image_url": data_url}
            candidate_parts = selected_parts + [label_part, image_part]
            estimated_request_bytes = self._estimate_qwen_input_bytes(
                [{"role": "user", "content": base_content + candidate_parts}]
            )
            item_debug["estimated_request_bytes_if_sent"] = estimated_request_bytes
            if estimated_request_bytes > QWEN_RESPONSES_REQUEST_BODY_SOFT_LIMIT_BYTES:
                # 软限制兜底的是整包大小：即使单图压缩成功，也不能让最终 JSON 请求接近 DashScope 硬上限。
                self._mark_qwen_image_skipped(item_debug, "request_body_soft_limit_exceeded")
                debug["images"].append(item_debug)
                continue

            item_debug["sent"] = True
            item_debug["skip_reason"] = ""
            selected_parts.extend([label_part, image_part])
            debug["sent_image_count"] += 1
            debug["images"].append(item_debug)

        debug["dropped_image_count"] = len([item for item in debug["images"] if not item.get("sent")])
        if debug["dropped_image_count"]:
            debug["fallback_to_asset_summary"] = True
        return selected_parts, debug

    @staticmethod
    def _build_qwen_image_label_text(image: Dict[str, Any]) -> str:
        return (
            "[Attached Image Evidence] "
            f"source_id={image.get('source_id', '')} "
            f"page={image.get('page_number', '')} "
            f"summary={image.get('asset_summary', '')} "
            f"section={image.get('section_path', '')}"
        )

    @staticmethod
    def _estimate_qwen_input_bytes(qwen_input: Any) -> int:
        # 使用紧凑 JSON 估算 Responses API 入参体积；软限制会额外预留模型名和 extra_body 的空间。
        return len(json.dumps(qwen_input, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))

    @staticmethod
    def _new_qwen_image_debug_item(image: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "source_id": str(image.get("source_id", "") or ""),
            "image_path": str(image.get("image_path", "") or ""),
            "page_number": image.get("page_number", ""),
            "sent": False,
            "skip_reason": "",
            "fallback_to_asset_summary": False,
        }

    @staticmethod
    def _mark_qwen_image_skipped(item_debug: Dict[str, Any], reason: str) -> None:
        item_debug["sent"] = False
        item_debug["skip_reason"] = reason
        item_debug["fallback_to_asset_summary"] = True

    def stream_qwen_responses(
        self,
        query: str,
        context: str,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        task_type: Optional[str] = None,
        enable_thinking: bool = QWEN_RERANK_COMPRESS_ENABLE_THINKING,
        image_inputs: Optional[List[Dict[str, Any]]] = None,
        asset_metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Stream a Qwen Responses API answer chunk by chunk."""
        try:
            if not api_key:
                api_key = QWEN_API_KEY
            if not api_key:
                raise ValueError("Qwen API key not provided")

            model_selection = self._resolve_qwen_model_selection(
                task_type=task_type,
                model_name=model_name,
                default_role="large",
            )
            model_name = model_selection["selected_model"]

            client = OpenAI(api_key=api_key, base_url=QWEN_BASE_URL)
            qwen_input, input_debug = self._build_qwen_input_with_debug(
                query=query,
                context=context,
                image_inputs=image_inputs,
                asset_metadata=asset_metadata,
            )
            stream = client.responses.create(
                model=model_name,
                input=qwen_input,
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
                        "qwen_request_debug": input_debug,
                    }
                    return

            yield {
                "type": "completed",
                "answer": "".join(answer_parts),
                "usage": None,
                "qwen_request_debug": input_debug,
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
        query: str,
        search_results: List[Dict],
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        show_reasoning: bool = True,
        image_inputs: Optional[List[Dict[str, Any]]] = None,
        asset_metadata: Optional[List[Dict[str, Any]]] = None,
        task_type: Optional[str] = None,
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

            model_selection: Dict[str, str] = {
                "task_type": self._normalize_task_type(task_type) or "default",
                "model_role": "custom" if model_name else "large",
                "selected_model": str(model_name or "").strip(),
                "routing_source": "explicit_model_name" if model_name else "task_route",
                "fallback_reason": "",
            }

            # Qwen 的模型选择收口到统一路由，避免各业务节点自己硬编码小/大模型名。
            if provider == "qwen":
                model_selection = self._resolve_qwen_model_selection(
                    task_type=task_type,
                    model_name=model_name,
                    default_role="large",
                )
                model_name = model_selection["selected_model"]

            # 根据不同提供商生成回答
            qwen_request_debug: Dict[str, Any] = {}
            if provider == "openai":
                if not model_name:
                    raise ValueError("OpenAI model name is required")
                response = self._generate_with_openai(model_name, query, context, api_key)
            elif provider == "qwen":
                response = self._generate_with_qwen_responses(
                    query,
                    context,
                    api_key,
                    model_name,
                    task_type=task_type,
                    enable_thinking=QWEN_RERANK_COMPRESS_ENABLE_THINKING,
                    image_inputs=image_inputs,
                    asset_metadata=asset_metadata,
                    request_debug=qwen_request_debug,
                )
            elif provider == "deepseek":
                if not model_name:
                    raise ValueError("DeepSeek model name is required")
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
                "task_type": model_selection.get("task_type", "default"),
                "selected_model": model_selection.get("selected_model", model_name or ""),
                "model_role": model_selection.get("model_role", ""),
                "routing_source": model_selection.get("routing_source", ""),
                "fallback_reason": model_selection.get("fallback_reason", ""),
                "response": response,
                "context": search_results,
                "image_inputs": image_inputs or [],
                "asset_metadata": asset_metadata or [],
                "qwen_request_debug": qwen_request_debug,
            }
            
            # 生成文件名并保存
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"generation_{provider}_{model_name or 'auto'}_{timestamp}.json"
            filepath = self.generation_results_dir / filename
            
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
                
            return {
                "response": response,
                "saved_filepath": str(filepath),
                "qwen_request_debug": qwen_request_debug,
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
            return self._image_bytes_to_data_url(image_file.read(), mime_type)

    def _prepare_qwen_image_data_url(self, image_path: str, *, target_bytes: int) -> Dict[str, Any]:
        if not image_path:
            raise ValueError("Image path is required for multimodal generation")
        if not os.path.exists(image_path):
            raise ValueError(f"Image path does not exist: {image_path}")

        mime_type = mimetypes.guess_type(image_path)[0] or "image/png"
        with open(image_path, "rb") as image_file:
            original_bytes = image_file.read()
        original_size = len(original_bytes)
        debug: Dict[str, Any] = {
            "original_bytes": original_size,
            "original_mime_type": mime_type,
            "compressed": False,
            "compressed_bytes": original_size,
            "output_mime_type": mime_type,
            "compression_attempts": [],
        }
        if original_size <= target_bytes:
            debug["data_url_bytes"] = len(self._image_bytes_to_data_url(original_bytes, mime_type).encode("utf-8"))
            return {
                "data_url": self._image_bytes_to_data_url(original_bytes, mime_type),
                "debug": debug,
            }

        try:
            from PIL import Image, ImageOps
        except ImportError:
            # Pillow 不可用时不能安全压缩大图，直接降级为图片摘要，避免原图 base64 触发 400。
            debug["skip_reason"] = "pillow_unavailable_for_large_image"
            return {"data_url": "", "debug": debug}

        best_bytes: Optional[bytes] = None
        best_attempt: Dict[str, Any] = {}
        try:
            with Image.open(io.BytesIO(original_bytes)) as opened:
                normalized = ImageOps.exif_transpose(opened)
                for max_side, quality in QWEN_RESPONSES_IMAGE_COMPRESSION_STEPS:
                    candidate = self._resize_qwen_image_for_jpeg(normalized, max_side=max_side, image_module=Image)
                    output = io.BytesIO()
                    candidate.save(output, format="JPEG", quality=quality, optimize=True)
                    compressed_bytes = output.getvalue()
                    attempt = {
                        "max_side": max_side,
                        "quality": quality,
                        "bytes": len(compressed_bytes),
                    }
                    debug["compression_attempts"].append(attempt)
                    if best_bytes is None or len(compressed_bytes) < len(best_bytes):
                        best_bytes = compressed_bytes
                        best_attempt = attempt
                    if len(compressed_bytes) <= target_bytes:
                        debug.update(
                            {
                                "compressed": True,
                                "compressed_bytes": len(compressed_bytes),
                                "output_mime_type": "image/jpeg",
                                "selected_compression": attempt,
                                "data_url_bytes": len(
                                    self._image_bytes_to_data_url(compressed_bytes, "image/jpeg").encode("utf-8")
                                ),
                            }
                        )
                        return {
                            "data_url": self._image_bytes_to_data_url(compressed_bytes, "image/jpeg"),
                            "debug": debug,
                        }
        except Exception as exc:
            # 解析或转码异常只影响图片输入；调用方会继续使用文本和图片摘要回答。
            debug["skip_reason"] = "image_compression_failed"
            debug["error"] = str(exc)
            return {"data_url": "", "debug": debug}

        debug["compressed"] = bool(best_bytes)
        debug["compressed_bytes"] = len(best_bytes or b"")
        debug["output_mime_type"] = "image/jpeg" if best_bytes else mime_type
        debug["selected_compression"] = best_attempt
        debug["skip_reason"] = "compressed_image_too_large"
        return {"data_url": "", "debug": debug}

    @staticmethod
    def _resize_qwen_image_for_jpeg(image: Any, *, max_side: int, image_module: Any) -> Any:
        width, height = image.size
        scale = min(1.0, float(max_side) / max(width, height))
        if scale < 1.0:
            resized_size = (max(1, int(width * scale)), max(1, int(height * scale)))
            resampling = getattr(getattr(image_module, "Resampling", image_module), "LANCZOS")
            image = image.resize(resized_size, resampling)

        # JPEG 没有透明通道；透明 PNG 使用白底合成，避免压缩后出现黑底或异常 alpha。
        if image.mode in {"RGBA", "LA"} or (image.mode == "P" and image.info.get("transparency") is not None):
            rgba = image.convert("RGBA")
            background = image_module.new("RGBA", rgba.size, (255, 255, 255, 255))
            background.alpha_composite(rgba)
            return background.convert("RGB")
        return image.convert("RGB")

    @staticmethod
    def _image_bytes_to_data_url(image_bytes: bytes, mime_type: str) -> str:
        encoded = base64.b64encode(image_bytes).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"
