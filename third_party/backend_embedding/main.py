import os
import time
import logging
import asyncio
from typing import List, Union, Optional, Literal
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import torch

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "voyageai/voyage-4-nano")
DEFAULT_DIMS = int(os.environ.get("NEO4J_VECTOR_DIMENSIONS", "2048"))
MAX_CONCURRENT_REQ = max(1, int(os.environ.get("EMBED_MAX_CONCURRENT_REQ", "1")))
IDLE_CACHE_CLEAR_SEC = float(os.environ.get("EMBED_IDLE_CACHE_CLEAR_SEC", "300"))
IDLE_CHECK_INTERVAL_SEC = float(os.environ.get("EMBED_IDLE_CHECK_INTERVAL_SEC", "30"))

# Device Selection
USE_CPU = os.environ.get("EMBEDDING_USE_CPU", "false").lower() in ("true", "1", "yes")
if USE_CPU:
    DEVICE = "cpu"
else:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Memory Management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Global Model
models = {}
encode_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQ)
request_state_lock = asyncio.Lock()
last_request_ts = time.time()
active_requests = 0
idle_cache_task: Optional[asyncio.Task] = None


async def _mark_request_started():
    global last_request_ts, active_requests
    async with request_state_lock:
        active_requests += 1
        last_request_ts = time.time()


async def _mark_request_finished():
    global last_request_ts, active_requests
    async with request_state_lock:
        active_requests = max(0, active_requests - 1)
        last_request_ts = time.time()


async def _idle_cache_monitor():
    """Clear CUDA allocator cache only when truly idle for a while."""
    global last_request_ts
    if DEVICE != "cuda" or not torch.cuda.is_available():
        return
    if IDLE_CACHE_CLEAR_SEC <= 0:
        return

    while True:
        await asyncio.sleep(max(1.0, IDLE_CHECK_INTERVAL_SEC))
        async with request_state_lock:
            idle_for = time.time() - last_request_ts
            is_busy = active_requests > 0

        if not is_busy and idle_for >= IDLE_CACHE_CLEAR_SEC:
            torch.cuda.empty_cache()
            logger.info(f"Cleared CUDA cache after idle for {idle_for:.1f}s")
            async with request_state_lock:
                last_request_ts = time.time()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global idle_cache_task
    # Load Model on Startup
    logger.info(f"Loading embedding model: {MODEL_NAME} on {DEVICE}...")
    logger.info(
        f"Embedding service limits: max_concurrent_req={MAX_CONCURRENT_REQ}, "
        f"idle_cache_clear_sec={IDLE_CACHE_CLEAR_SEC}, idle_check_interval_sec={IDLE_CHECK_INTERVAL_SEC}"
    )
    try:
        model_instance = SentenceTransformer(
            MODEL_NAME, 
            trust_remote_code=True,
            device=DEVICE,
            truncate_dim=DEFAULT_DIMS
        )
        model_instance.max_seq_length = 8192
        model_instance.eval()
        

        models["model"] = model_instance
        logger.info(f"Model loaded successfully")

        if DEVICE == "cuda" and torch.cuda.is_available() and IDLE_CACHE_CLEAR_SEC > 0:
            idle_cache_task = asyncio.create_task(_idle_cache_monitor())
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise e
    yield
    if idle_cache_task:
        idle_cache_task.cancel()
        with suppress(asyncio.CancelledError):
            await idle_cache_task
        idle_cache_task = None
    models.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(title="Embedding Service", lifespan=lifespan)

class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: Optional[str] = "embedding-model"
    encoding_format: Optional[str] = "float"
    
    # Custom Extensions
    encoding_type: Literal["query", "document"] = "document"
    dimensions: Optional[int] = DEFAULT_DIMS
    
    class Config:
        extra = "allow"

class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[dict]
    model: str
    usage: dict

def _sync_encode(model, texts, encoding_type):
    """Heavy computation moved to thread."""
    with torch.no_grad():
        if encoding_type == "query":
            if hasattr(model, "encode_query"):
                return model.encode_query(texts, convert_to_numpy=True)
            else:
                return model.encode(texts, convert_to_numpy=True, prompt_name="query") 
        else:
            if hasattr(model, "encode_document"):
                 return model.encode_document(texts, convert_to_numpy=True)
            else:
                 return model.encode(texts, convert_to_numpy=True)

@app.post("/v1/embeddings")
async def create_embeddings(req: EmbeddingRequest):
    model = models.get("model")
    if not model:
        raise HTTPException(status_code=500, detail="Model not initialized")

    # input normalization
    texts = req.input
    if isinstance(texts, str):
        texts = [texts]

    start_time = time.time()
    
    try:
        started = False
        async with encode_semaphore:
            await _mark_request_started()
            started = True
            try:
                # Offload blocking GPU call to a thread pool
                raw_embeds = await asyncio.to_thread(_sync_encode, model, texts, req.encoding_type)
            finally:
                if started:
                    await _mark_request_finished()
        
        # Return Raw Vector
        if hasattr(raw_embeds, "tolist"):
            raw_embeds = raw_embeds.tolist()
            
        data = [
            {
                "object": "embedding",
                "embedding": emb,
                "index": i
            }
            for i, emb in enumerate(raw_embeds)
        ]
        
        elapsed = time.time() - start_time
        logger.info(f"Embedding generated for {len(texts)} texts in {elapsed:.4f} seconds (Device: {DEVICE})")

        return EmbeddingResponse(
            data=data,
            model=MODEL_NAME,
            usage={
                "prompt_tokens": 0, 
                "total_tokens": 0
            }
        )

    except Exception as e:
        logger.error(f"Embedding error: {e}")
        # OOM 발생 시 긴급 캐시 정리
        if "CUDA out of memory" in str(e):
            torch.cuda.empty_cache()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}

# --- Tokenizer Endpoints (vLLM Compatibility) ---
class TokenizeRequest(BaseModel):
    model: str = "embedding-model"
    prompt: str

class DetokenizeRequest(BaseModel):
    model: str = "embedding-model"
    tokens: List[int]

@app.post("/v1/tokenize")
async def tokenize(req: TokenizeRequest):
    model = models.get("model")
    if not model or not hasattr(model, "tokenizer"):
        raise HTTPException(status_code=500, detail="Tokenizer not available")
    
    # SentenceTransformer uses internal tokenizer
    # Access underlying tokenizer
    tokenizer = model.tokenizer
    
    # Tokenize
    # ST tokenizer usually returns dict or list. We want Int IDs.
    # tokenizer.encode(text) -> [ids]
    token_ids = tokenizer.encode(req.prompt, add_special_tokens=False)
    
    return {
        "tokens": token_ids,
        "count": len(token_ids),
        "max_model_len": model.max_seq_length
    }

@app.post("/v1/detokenize")
async def detokenize(req: DetokenizeRequest):
    model = models.get("model")
    if not model or not hasattr(model, "tokenizer"):
        raise HTTPException(status_code=500, detail="Tokenizer not available")
        
    tokenizer = model.tokenizer
    
    # Detokenize
    text = tokenizer.decode(req.tokens, skip_special_tokens=True)
    
    return {
        "prompt": text
    }
