import os
import logging
import asyncio
import re
import copy
import json
import inspect
import httpx
import openai
import tiktoken
from openai import AsyncOpenAI
from typing import List, Dict, Any, Optional
from utils.parsers import clean_and_unwrap_json
from .config import RAGConfig

class VLLMClient:
    _client_cache = {}

    def __init__(self, model_name: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.vllm_url = RAGConfig.VLLM_URL
        self.ocr_url = RAGConfig.VLLM_OCR_URL
        self.embed_url = RAGConfig.VLLM_EMBED_URL
        self.rerank_url = RAGConfig.VLLM_RERANK_URL
        
        self.model_name = model_name or RAGConfig.DEFAULT_MODEL
        self.ocr_model_name = RAGConfig.OCR_MODEL
        self.embed_model_name = RAGConfig.EMBEDDING_MODEL

        self.api_key = os.environ.get("VLLM_API_KEY", "EMPTY")
        # 0 = infinite timeout (None)
        timeout_val = RAGConfig.LLM_REQUEST_TIMEOUT
        self._request_timeout = None if timeout_val == 0 else timeout_val
        self._embed_semaphore = asyncio.Semaphore(RAGConfig.MAX_CONCURRENT_LLM_CALLS)
        self._embed_accepts_encoding_type: Optional[bool] = None
        self._embed_server_max_len: Optional[int] = None
        self._embed_server_max_len_checked: bool = False

        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.tokenizer = None

    def _count_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Rough token count for OpenAI messages."""
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                if key == "content":
                    if isinstance(value, list):
                        for item in value:
                            if item.get("type") == "text":
                                content = item.get("text", "")
                                if self.tokenizer:
                                    num_tokens += len(self.tokenizer.encode(content))
                                else:
                                    num_tokens += len(content) // 4
                            elif item.get("type") == "image_url":
                                num_tokens += 85 # rough estimate for image
                    else:
                        content = str(value)
                        if self.tokenizer:
                            num_tokens += len(self.tokenizer.encode(content))
                        else:
                            num_tokens += len(content) // 4
                if key == "name":
                    num_tokens += 1 # role is always 1 token, name adds 1
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens

    def _truncate_messages(self, messages: List[Dict[str, Any]], max_tokens: int = RAGConfig.MAX_CONTEXT_LENGTH) -> List[Dict[str, Any]]:
        """
        Truncates messages to fit within max_tokens.
        Strategy: Keep system message and most recent messages.
        """
        # Reserve tokens for completion (e.g., 1024)
        effective_limit = max_tokens - 1024
        
        if self._count_tokens(messages) <= effective_limit:
            return messages

        self.logger.warning(f"Messages too long ({self._count_tokens(messages)} tokens). Truncating to {effective_limit}...")

        # 1. Keep system message if it exists
        system_msg = None
        if messages and messages[0].get("role") == "system":
            system_msg = messages[0]
            messages = messages[1:]
        
        # 2. Add messages from the end until limit is reached
        truncated = []
        current_tokens = 0
        if system_msg:
            current_tokens = self._count_tokens([system_msg])
            
        for msg in reversed(messages):
            msg_tokens = self._count_tokens([msg])
            if current_tokens + msg_tokens <= effective_limit:
                truncated.insert(0, msg)
                current_tokens += msg_tokens
            else:
                # If even one message is too long, we might need to truncate its content
                if not truncated:
                     # For the last message (which is actually the most recent user prompt), 
                     # we try to keep as much as possible
                     content = msg.get("content", "")
                     if isinstance(content, str):
                         allowed = effective_limit - current_tokens
                         if allowed > 100:
                             # Very rough character-based truncation
                             msg["content"] = content[:allowed * 3] + "...(truncated)"
                             truncated.insert(0, msg)
                break
                
        if system_msg:
            truncated.insert(0, system_msg)
            
        return truncated

    def _truncate_text(self, text: str, max_tokens: int = RAGConfig.MAX_EMBEDDING_LENGTH) -> str:
        """Truncates a single string to fit within max_tokens."""
        if not text: return ""
        if self.tokenizer:
            tokens = self.tokenizer.encode(text)
            if len(tokens) <= max_tokens:
                return text
            return self.tokenizer.decode(tokens[:max_tokens])
        else:
            # Fallback to rough character count
            char_limit = max_tokens * 3
            if len(text) <= char_limit:
                return text
            return text[:char_limit]

    @staticmethod
    def _parse_positive_int(value: Any) -> Optional[int]:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value if value > 0 else None
        if isinstance(value, float):
            value = int(value)
            return value if value > 0 else None
        if isinstance(value, str):
            value = value.strip()
            if value.isdigit():
                parsed = int(value)
                return parsed if parsed > 0 else None
        return None

    def _resolve_output_token_limit(self, requested_max_tokens: Any = None) -> int:
        cap = max(1, int(RAGConfig.MAX_OUTPUT_TOKENS))
        requested = self._parse_positive_int(requested_max_tokens)
        if requested is None:
            return cap
        return min(requested, cap)

    @staticmethod
    def _prefers_max_completion_tokens(model: Any) -> bool:
        name = str(model or "").strip().lower()
        if not name:
            return False
        # GPT-5 / o-series models require max_completion_tokens.
        return (
            name.startswith("gpt-5")
            or name.startswith("o1")
            or name.startswith("o3")
            or name.startswith("o4")
        )

    @staticmethod
    def _json_error_context(text: Any, pos: Optional[int], radius: int = 120) -> str:
        raw = str(text or "")
        if not raw:
            return ""

        if pos is None or pos < 0:
            excerpt = raw[: max(1, radius * 2)]
            return excerpt.replace("\n", "\\n").replace("\r", "\\r")

        idx = min(max(int(pos), 0), max(0, len(raw) - 1))
        start = max(0, idx - radius)
        end = min(len(raw), idx + radius)
        excerpt = raw[start:end]
        marker = idx - start
        if 0 <= marker < len(excerpt):
            excerpt = (
                excerpt[:marker]
                + "<<<ERR>>>"
                + excerpt[marker]
                + "<<<ERR>>>"
                + excerpt[marker + 1:]
            )
        return excerpt.replace("\n", "\\n").replace("\r", "\\r")

    def _extract_embed_server_max_len(self, payload: Any) -> Optional[int]:
        if not isinstance(payload, dict):
            return None

        data = payload.get("data")
        if not isinstance(data, list):
            return None

        def _find_max_len(entries: List[Any]) -> Optional[int]:
            for entry in entries:
                if not isinstance(entry, dict):
                    continue

                for key in ("max_model_len", "max_seq_len", "max_length", "context_length"):
                    parsed = self._parse_positive_int(entry.get(key))
                    if parsed is not None:
                        return parsed

                for meta_key in ("model_info", "metadata", "extra"):
                    meta = entry.get(meta_key)
                    if not isinstance(meta, dict):
                        continue
                    for key in ("max_model_len", "max_seq_len", "max_length", "context_length"):
                        parsed = self._parse_positive_int(meta.get(key))
                        if parsed is not None:
                            return parsed
            return None

        target_model_name = (self.embed_model_name or "").strip()
        matched: List[Any] = []
        others: List[Any] = []
        for entry in data:
            if isinstance(entry, dict) and str(entry.get("id", "")).strip() == target_model_name:
                matched.append(entry)
            else:
                others.append(entry)

        return _find_max_len(matched) or _find_max_len(others)

    async def _refresh_embed_server_max_len(self):
        if self._embed_server_max_len_checked:
            return
        self._embed_server_max_len_checked = True

        if os.environ.get("RAG_EMBED_DISCOVER_MAX_LEN", "false").lower() != "true":
            self.logger.debug(
                "Embedding max_len auto-discovery skipped: RAG_EMBED_DISCOVER_MAX_LEN is false."
            )
            return

        if not self.api_key or self.api_key == "EMPTY":
            self.logger.debug(
                "Embedding max_len auto-discovery skipped: VLLM_API_KEY is not set."
            )
            return

        url = f"{self.embed_url.rstrip('/')}/models"
        headers: Dict[str, str] = {}
        if self.api_key and self.api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            async with httpx.AsyncClient(timeout=self._request_timeout) as client:
                response = await client.get(url, headers=headers)

            if response.status_code != 200:
                self.logger.debug(
                    "Embedding max_len auto-discovery skipped: GET %s returned %s",
                    url,
                    response.status_code,
                )
                return

            discovered = self._extract_embed_server_max_len(response.json())
            if discovered is not None:
                self._embed_server_max_len = discovered
                self.logger.info(
                    "Embedding max_len discovered from server: %d tokens",
                    discovered,
                )
        except Exception as e:
            self.logger.debug("Embedding max_len auto-discovery failed: %s", e)

    def _embedding_token_limit(self, aggressive: bool = False, base_max_tokens: Optional[int] = None) -> int:
        """
        Return a conservative embedding token limit with safety reserve.
        This avoids off-by-one and tokenizer-mismatch overflows at provider side.
        """
        effective_max = (
            base_max_tokens
            if base_max_tokens is not None
            else (self._embed_server_max_len or RAGConfig.MAX_EMBEDDING_LENGTH)
        )
        reserve = int(os.environ.get("RAG_EMBEDDING_TOKEN_RESERVE", "0"))
        base = max(256, effective_max - max(0, reserve))
        if aggressive:
            return max(128, int(base * 0.75))
        return base

    @staticmethod
    def _is_context_length_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return (
            "maximum context length" in msg
            or "input tokens" in msg
            or "too many tokens" in msg
        )

    def _truncate_for_rerank(
        self,
        query: str,
        documents: List[str],
        doc_max_tokens: Optional[int] = None
    ) -> tuple[str, List[str]]:
        safe_query = self._truncate_text(
            str(query or ""),
            max_tokens=max(16, RAGConfig.RERANK_QUERY_MAX_TOKENS),
        )
        max_doc_tokens = max(
            64,
            doc_max_tokens if doc_max_tokens is not None else RAGConfig.RERANK_DOC_MAX_TOKENS,
        )
        safe_documents = [
            self._truncate_text(str(doc or ""), max_tokens=max_doc_tokens)
            for doc in documents
        ]
        return safe_query, safe_documents

    def _is_qwen_embedding_model(self) -> bool:
        return "qwen3-embedding" in (self.embed_model_name or "").lower()

    def _format_query_for_embedding(self, query: str) -> str:
        """Apply model-recommended query instruction format for Qwen embedding models."""
        task = os.environ.get(
            "EMBEDDING_QUERY_INSTRUCTION",
            "Given a web search query, retrieve relevant passages that answer the query",
        )
        return f"Instruct: {task}\nQuery:{query}"

    async def _create_embedding_request(self, inputs: List[str], encoding_type: str):
        request_kwargs: Dict[str, Any] = {
            "model": self.embed_model_name,
            "input": inputs
        }
        use_encoding_hint = (
            os.environ.get("RAG_EMBED_SEND_ENCODING_TYPE", "false").lower() == "true"
            and self._embed_accepts_encoding_type is not False
        )
        if use_encoding_hint:
            request_kwargs["extra_body"] = {"encoding_type": encoding_type}

        try:
            async with self._embed_semaphore:
                response = await self._retry_with_backoff(
                    self.embed_client.embeddings.create,
                    **request_kwargs
                )
            if use_encoding_hint and self._embed_accepts_encoding_type is None:
                self._embed_accepts_encoding_type = True
            return response
        except Exception as e:
            # Some embedding backends reject custom fields in payload.
            msg = str(e).lower()
            hint_rejected = any(k in msg for k in [
                "encoding_type",
                "extra_body",
                "unexpected",
                "validation",
                "unrecognized",
                "unknown field"
            ])
            if use_encoding_hint and hint_rejected:
                self.logger.warning(
                    "Embedding endpoint rejected encoding_type hint. Retrying without it."
                )
                self._embed_accepts_encoding_type = False
                fallback_kwargs = {
                    "model": self.embed_model_name,
                    "input": inputs
                }
                async with self._embed_semaphore:
                    return await self._retry_with_backoff(
                        self.embed_client.embeddings.create,
                        **fallback_kwargs
                    )
            raise e

    async def _embed_batch_itemwise(self, batch: List[str], encoding_type: str) -> List[List[float]]:
        embeddings: List[List[float]] = []
        for idx, text in enumerate(batch):
            try:
                response = await self._create_embedding_request([text], encoding_type=encoding_type)
                if getattr(response, "data", None):
                    embeddings.append(response.data[0].embedding)
                else:
                    self.logger.error("Embedding item response missing data at idx=%d.", idx)
                    embeddings.append([])
            except Exception as e:
                recovered = False
                if self._is_context_length_error(e):
                    aggressive_text = self._truncate_text(
                        text,
                        max_tokens=self._embedding_token_limit(aggressive=True)
                    )
                    if aggressive_text and aggressive_text != text:
                        try:
                            response = await self._create_embedding_request([aggressive_text], encoding_type=encoding_type)
                            if getattr(response, "data", None):
                                self.logger.warning(
                                    "Embedding item recovered with aggressive truncation at idx=%d.",
                                    idx,
                                )
                                embeddings.append(response.data[0].embedding)
                                recovered = True
                        except Exception as e2:
                            self.logger.error(
                                "Embedding item aggressive retry failed at idx=%d: %s",
                                idx,
                                e2,
                            )
                if not recovered:
                    preview = text.replace("\n", " ")[:120]
                    self.logger.error("Embedding item failed at idx=%d text='%s': %s", idx, preview, e)
                    embeddings.append([])
        return embeddings

    async def rerank(self, query: str, documents: List[str], instruction: Optional[str] = None) -> List[float]:
        """Calls the specialized reranker service."""
        if not documents: return []
        original_count = len(documents)
        safe_query, safe_documents = self._truncate_for_rerank(query, documents)
        batch_size = max(1, RAGConfig.RERANK_BATCH_SIZE)

        def _normalize_scores(raw_scores: Any, expected_count: int) -> List[float]:
            if not isinstance(raw_scores, list):
                return [0.0] * expected_count
            scores: List[float] = []
            for value in raw_scores[:expected_count]:
                if isinstance(value, (int, float)):
                    scores.append(float(value))
                else:
                    scores.append(0.0)
            if len(scores) < expected_count:
                scores.extend([0.0] * (expected_count - len(scores)))
            return scores

        try:
            # Base URL is expected to be something like http://localhost:18083/v1
            url = f"{self.rerank_url.rstrip('/')}/rerank"
            
            timeout = None if RAGConfig.LLM_REQUEST_TIMEOUT == 0 else RAGConfig.LLM_REQUEST_TIMEOUT
            async def _call_rerank(rerank_query: str, rerank_documents: List[str]):
                payload = {
                    "query": rerank_query,
                    "documents": rerank_documents
                }
                if instruction:
                    payload["instruction"] = instruction
                async with httpx.AsyncClient(timeout=timeout) as client:
                    return await client.post(url, json=payload)

            async def _score_batch(batch_documents: List[str]) -> List[float]:
                response = await self._retry_with_backoff(_call_rerank, safe_query, batch_documents)
                if response.status_code == 200:
                    return _normalize_scores(response.json().get("scores", []), len(batch_documents))

                self.logger.error("Rerank failed: %s - %s", response.status_code, response.text)
                if response.status_code >= 500:
                    fallback_query, fallback_documents = self._truncate_for_rerank(
                        safe_query,
                        batch_documents,
                        doc_max_tokens=RAGConfig.RERANK_OVERFLOW_DOC_MAX_TOKENS
                    )
                    fallback_response = await self._retry_with_backoff(
                        _call_rerank,
                        fallback_query,
                        fallback_documents
                    )
                    if fallback_response.status_code == 200:
                        self.logger.warning(
                            "Reranker recovered with aggressive truncation (doc_max_tokens=%d).",
                            RAGConfig.RERANK_OVERFLOW_DOC_MAX_TOKENS,
                        )
                        return _normalize_scores(fallback_response.json().get("scores", []), len(batch_documents))
                    self.logger.error(
                        "Rerank fallback failed: %s - %s",
                        fallback_response.status_code,
                        fallback_response.text
                    )
                return [0.0] * len(batch_documents)

            if original_count <= batch_size:
                return await _score_batch(safe_documents)

            self.logger.info(
                "Reranking in batches: docs=%d batch_size=%d",
                original_count,
                batch_size,
            )
            all_scores: List[float] = []
            for start in range(0, original_count, batch_size):
                end = min(start + batch_size, original_count)
                batch_scores = await _score_batch(safe_documents[start:end])
                all_scores.extend(batch_scores)
            if len(all_scores) < original_count:
                all_scores.extend([0.0] * (original_count - len(all_scores)))
            return all_scores[:original_count]
        except Exception as e:
            self.logger.error(f"Error calling Reranker: {e}")
            return [0.0] * original_count

    async def _retry_with_backoff(self, coro_func, *args, **kwargs):
        """Exponential backoff retry wrapper for handling GPU load spikes."""
        for attempt in range(RAGConfig.LLM_MAX_RETRIES):
            try:
                return await coro_func(*args, **kwargs)
            except (httpx.TimeoutException, openai.APITimeoutError) as e:
                if attempt == RAGConfig.LLM_MAX_RETRIES - 1:
                    raise
                delay = RAGConfig.LLM_RETRY_DELAY * (2 ** attempt)
                self.logger.warning(f"Timeout, retrying in {delay}s... ({attempt+1}/{RAGConfig.LLM_MAX_RETRIES})")
                await asyncio.sleep(delay)

    def _get_cached_client(self, url: str) -> AsyncOpenAI:
        if url not in self._client_cache:
            timeout = httpx.Timeout(self._request_timeout, connect=60.0)
            self._client_cache[url] = AsyncOpenAI(base_url=url, api_key=self.api_key, timeout=timeout)
        return self._client_cache[url]

    @property
    def client(self): return self._get_cached_client(self.vllm_url)
    @property
    def ocr_client(self): return self._get_cached_client(self.ocr_url)
    @property
    def embed_client(self): return self._get_cached_client(self.embed_url)

    @property
    def judge_client(self):
        """Unified client for LLM-as-a-judge (uses OpenAI if API key exists, else local vLLM)."""
        if RAGConfig.OPENAI_API_KEY:
            # Official OpenAI
            if "openai_official" not in self._client_cache:
                self._client_cache["openai_official"] = AsyncOpenAI(api_key=RAGConfig.OPENAI_API_KEY)
            return self._client_cache["openai_official"]
        else:
            # Fallback to local vLLM
            return self.client

    @staticmethod
    def _is_openai_model(model: str) -> bool:
        """Return True if the model name should be routed to the OpenAI API."""
        if not model:
            return False
        _OPENAI_PREFIXES = ("gpt-", "o1", "o3", "o4", "chatgpt-")
        return any(model.lower().startswith(p) for p in _OPENAI_PREFIXES)

    def think_strip(self, message: Optional[str]) -> str:
        if not message: return ""
        if "</think>" in message:
            message = message.split("</think>")[-1]
        return message.replace("<end>", "").strip()

    async def generate_with_image(self, image_base64: str, prompt: str = "", system_prompt: Optional[str] = None) -> str:
        """
        Vision Language Model OCR using dedicated OCR service.
        
        Args:
            image_base64: Base64 encoded image string
            prompt: Optional text prompt (empty for pure OCR)
            system_prompt: Optional system prompt
            
        Returns:
            OCR extracted text
        """
        image_url = f"data:image/png;base64,{image_base64}"
        
        # Build messages with specific order: Image first, then Text instructions
        content = [{"type": "image_url", "image_url": {"url": image_url}}]
        if prompt:
            content.append({"type": "text", "text": prompt})
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})
        
        try:
            # Use dedicated OCR client with LightOnOCR compatible parameters
            ocr_params: Dict[str, Any] = {
                "model": self.ocr_model_name,
                "messages": messages,
                "stream": False,
                "temperature": RAGConfig.OCR_TEMPERATURE,
                "top_p": RAGConfig.OCR_TOP_P,
            }
            if os.environ.get("RAG_OCR_SEND_TOP_K", "false").lower() == "true":
                ocr_params["extra_body"] = {
                    "top_k": int(os.environ.get("RAG_OCR_TOP_K", "0"))
                }

            response = await self.ocr_client.chat.completions.create(**ocr_params)
            return self.think_strip(response.choices[0].message.content or "")
        except (httpx.TimeoutException, openai.APITimeoutError):
            self.logger.error("OCR Request Timeout: The OCR model took too long to respond.")
            raise Exception("OCR service timed out. Please try again.")
        except Exception as e:
            self.logger.error(f"Error calling OCR vLLM: {e}")
            raise

    async def generate_response(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None, tool_choice: Optional[str] = None, temperature: Optional[float] = None, **kwargs) -> Any:
        try:
            apply_default_sampling = bool(kwargs.pop("apply_default_sampling", True))
            # Truncate messages to fit context window
            truncated_messages = self._truncate_messages(messages)
            requested_max_tokens = kwargs.get("max_tokens", kwargs.get("max_completion_tokens"))
            
            params = {
                "model": kwargs.get("model", self.model_name),
                "messages": truncated_messages,
                "stream": False,
                "max_tokens": self._resolve_output_token_limit(requested_max_tokens),
            }
            if temperature is not None:
                params["temperature"] = temperature
            elif apply_default_sampling:
                params["temperature"] = 0.7
            if tools: params["tools"] = tools
            if tool_choice: params["tool_choice"] = tool_choice
            if kwargs.get("response_format"): params["response_format"] = kwargs["response_format"]
            params["extra_body"] = (
                kwargs["extra_body"]
                if "extra_body" in kwargs and kwargs["extra_body"] is not None
                else {"chat_template_kwargs": {"enable_thinking": False}}
            )

            _model_name = params.get("model", "")
            if self._is_openai_model(_model_name):
                params.pop("extra_body", None)
                _client = self.judge_client
            else:
                _client = self.client
            response = await self._retry_with_backoff(_client.chat.completions.create, **params)
            msg = response.choices[0].message
            if hasattr(msg, 'tool_calls') and msg.tool_calls: return msg
            
            content = msg.content or (msg.reasoning_content if hasattr(msg, "reasoning_content") else "")
            
            # If JSON format was requested, don't attempt to unwrap common keys
            if kwargs.get("response_format"):
                return self.think_strip(content)
                
            return self.think_strip(clean_and_unwrap_json(content))
        except Exception as e:
            self.logger.error(f"Error calling vLLM: {e}")
            raise e

    async def generate_json(self, messages: List[Dict[str, str]], max_retries: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        last_error_hint = ""
        max_retries = max_retries or RAGConfig.RETRY_COUNT
        json_debug_label = str(kwargs.pop("json_debug_label", "") or "").strip()
        for attempt in range(max_retries):
            current_messages = copy.deepcopy(messages)
            if last_error_hint:
                current_messages.append({"role": "user", "content": f"SYSTEM: {last_error_hint}"})
            response_text = ""
            try:
                # Route to eval_json if model is EVAL_MODEL or any OpenAI model
                model = kwargs.get("model")
                if model and (model == RAGConfig.EVAL_MODEL or self._is_openai_model(model)):
                    return await self.generate_eval_json(current_messages, model=model)
                
                response_text = await self.generate_response(current_messages, response_format={"type": "json_object"}, **kwargs)
                parsed = json.loads(response_text)
                if isinstance(parsed, dict):
                    return parsed
                last_error_hint = (
                    f"Invalid JSON type '{type(parsed).__name__}'. "
                    "Output ONLY one JSON object (not array/string/markdown)."
                )
            except json.JSONDecodeError as e:
                snippet = self._json_error_context(response_text, e.pos)
                self.logger.warning(
                    "generate_json parse failed [stage=%s] (attempt %d/%d): %s | len=%d pos=%d line=%d col=%d | snippet=%s",
                    json_debug_label or "unknown",
                    attempt + 1,
                    max_retries,
                    e,
                    len(response_text or ""),
                    e.pos,
                    e.lineno,
                    e.colno,
                    snippet,
                )
                last_error_hint = (
                    f"Invalid JSON near line {e.lineno}, column {e.colno}. "
                    "Output ONLY one raw JSON object with double quotes, no markdown fences, no prose."
                )
            except Exception as e:
                self.logger.warning(
                    "generate_json parse failed (attempt %d/%d): %s",
                    attempt + 1,
                    max_retries,
                    e,
                )
                last_error_hint = (
                    "Invalid JSON. Output ONLY one raw JSON object with double quotes, "
                    "no markdown fences, no prose."
                )
        return {}
    
    async def generate_eval_json(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Generate JSON using judge_client (OpenAI if API key exists, else local vLLM)."""
        model = kwargs.get("model", RAGConfig.EVAL_MODEL)
        try:
            # Truncate messages to fit context window
            truncated_messages = self._truncate_messages(messages)
            requested_max_tokens = kwargs.get("max_tokens", kwargs.get("max_completion_tokens"))

            token_limit = self._resolve_output_token_limit(requested_max_tokens)
            token_fields = ["max_tokens", "max_completion_tokens"]
            if self._prefers_max_completion_tokens(model):
                token_fields = ["max_completion_tokens", "max_tokens"]

            last_error: Optional[Exception] = None
            for token_field in token_fields:
                params: Dict[str, Any] = {
                    "model": model,
                    "messages": truncated_messages,
                    "response_format": {"type": "json_object"},
                    token_field: token_limit,
                }
                # gpt-5-nano and some newer OpenAI models only accept temperature=1 (default).
                # Omit temperature for OpenAI models to use their default.
                if not self._is_openai_model(model):
                    params["temperature"] = 0.0
                if kwargs.get("extra_body") is not None:
                    params["extra_body"] = kwargs["extra_body"]
                elif not self._is_openai_model(model):
                    # chat_template_kwargs is vLLM-specific; never send to OpenAI API
                    params["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
                try:
                    response = await self.judge_client.chat.completions.create(**params)
                    content = response.choices[0].message.content or ""
                    try:
                        parsed = json.loads(content)
                    except json.JSONDecodeError as e:
                        snippet = self._json_error_context(content, e.pos)
                        self.logger.warning(
                            "generate_eval_json parse failed (token_field=%s): %s | len=%d pos=%d line=%d col=%d | snippet=%s",
                            token_field,
                            e,
                            len(content or ""),
                            e.pos,
                            e.lineno,
                            e.colno,
                            snippet,
                        )
                        raise
                    return parsed if isinstance(parsed, dict) else {}
                except Exception as e:
                    last_error = e
                    msg = str(e).lower()
                    token_param_error = (
                        ("unsupported parameter" in msg or "not supported" in msg)
                        and ("max_tokens" in msg or "max_completion_tokens" in msg)
                    )
                    if token_param_error:
                        continue
                    raise
            if last_error is not None:
                raise last_error
            return {}
        except Exception as e:
            self.logger.error(f"Error calling evaluation LLM ({model}): {e}")
            raise

    async def get_embeddings(self, texts: List[str], encoding_type: str = "document") -> List[List[float]]:
        if not texts: return []

        await self._refresh_embed_server_max_len()

        # Truncate and format query texts to prevent embedding model overflow.
        embed_max_tokens = self._embedding_token_limit(base_max_tokens=self._embed_server_max_len)
        truncated_texts: List[str] = []
        for t in texts:
            candidate = self._truncate_text(t, max_tokens=embed_max_tokens)
            if encoding_type == "query" and self._is_qwen_embedding_model():
                candidate = self._format_query_for_embedding(candidate)
            truncated_texts.append(self._truncate_text(candidate, max_tokens=embed_max_tokens))
        
        all_embeddings = []
        for i in range(0, len(truncated_texts), RAGConfig.EMBEDDING_BATCH_SIZE):
            batch = truncated_texts[i:i + RAGConfig.EMBEDDING_BATCH_SIZE]
            try:
                response = await self._create_embedding_request(batch, encoding_type=encoding_type)
                all_embeddings.extend([item.embedding for item in response.data])
            except Exception as e:
                self.logger.error(
                    "Embedding batch failed (size=%d, type=%s): %s. Falling back to item-wise retry.",
                    len(batch),
                    encoding_type,
                    e,
                )
                all_embeddings.extend(await self._embed_batch_itemwise(batch, encoding_type=encoding_type))
        return all_embeddings

    async def get_embedding(self, text: str) -> List[float]:
        res = await self.get_embeddings([text], encoding_type="query")
        if not res or not res[0]:
            self.logger.warning("Failed to generate query embedding (empty vector).")
            return []
        return res[0]

    async def generate_queued(self, messages: List[Dict[str, str]], **kwargs):
        # Fallback to direct call since Celery is removed
        return await self.generate_response(messages, **kwargs)

    @classmethod
    async def global_close(cls):
        """Close cached API clients and clear cache."""
        logger = logging.getLogger(__name__)
        for key, client in list(cls._client_cache.items()):
            try:
                close_fn = getattr(client, "close", None)
                if callable(close_fn):
                    result = close_fn()
                    if inspect.isawaitable(result):
                        await result
                else:
                    aclose_fn = getattr(client, "aclose", None)
                    if callable(aclose_fn):
                        result = aclose_fn()
                        if inspect.isawaitable(result):
                            await result
            except Exception as e:
                logger.warning(f"Failed to close client cache entry '{key}': {e}")
        cls._client_cache.clear()

def get_llm_client(model_id: str = "local"):
    return VLLMClient(model_name=None if model_id == "local" else model_id)
