"""Official MS GraphRAG indexing wired to local vLLM.

Builds a GraphRagConfig that points LiteLLM at our local vLLM endpoints
(:28000 generation, :18082 embedding) and runs the standard pipeline
(extract_graph → Leiden communities → community reports → embeddings).

Outputs parquet under data/ms_graphrag_output/<corpus_tag>/. The query-time
adapter reads these parquet files instead of expecting Neo4j Community nodes.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger("HypoReflect")


# vLLM endpoints — fixed by run_servers.sh. gen=28000 (GPU 1), gen2=28010 (GPU 0).
# Default round-robins across both so MS uses both GPUs; primary base (passed
# as ModelConfig.api_base) is the first entry, but the LiteLLM Router monkey-
# patch below intercepts and shuffles across all bases per call.
_GEN_API_BASES = [
    s.strip() for s in os.environ.get(
        "RAG_MS_GEN_API_BASES",
        "http://localhost:28000/v1,http://localhost:28010/v1",
    ).split(",") if s.strip()
]
_GEN_API_BASE = _GEN_API_BASES[0]
_GEN_MODEL_NAME = os.environ.get("VLLM_SERVED_MODEL_NAME", "generation-model")
_EMBED_API_BASE = os.environ.get("RAG_MS_EMBED_API_BASE", "http://localhost:18082/v1")
_EMBED_MODEL_NAME = os.environ.get("RAG_MS_EMBED_MODEL_NAME", "embedding-model")

# Where parquet artifacts land. corpus_tag-scoped so different runs don't clobber.
_OUTPUT_ROOT = Path(os.environ.get("RAG_MS_OUTPUT_ROOT", "data/ms_graphrag_output"))


def output_dir_for(corpus_tag: str) -> Path:
    return (_OUTPUT_ROOT / corpus_tag).resolve()


def cache_dir_for(corpus_tag: str) -> Path:
    return (_OUTPUT_ROOT / corpus_tag / "_cache").resolve()


def input_dir_for(corpus_tag: str) -> Path:
    return (_OUTPUT_ROOT / corpus_tag / "_input").resolve()


def _stage_input_files(
    dataset_path: str,
    corpus_tag: str,
    sample_companies: Optional[list[str]],
) -> Path:
    """Copy/link selected files into a tag-scoped input dir.

    MS pipeline reads from one directory via input_storage.base_dir. We can't
    pass a file list, so we materialize a filtered staging dir under the
    output tree (hardlinks to avoid disk waste; falls back to copy on FS that
    rejects hardlinks).
    """
    import json

    src_root = Path(dataset_path)
    if not src_root.exists():
        raise FileNotFoundError(f"dataset_path not found: {dataset_path}")

    files = sorted(p for p in src_root.iterdir() if p.suffix in (".txt", ".md"))

    if sample_companies:
        doc_info_path = Path("data/financebench_document_information.jsonl")
        if doc_info_path.exists():
            with doc_info_path.open() as fh:
                doc_data = [json.loads(line) for line in fh]
            valid = {item["doc_name"] for item in doc_data if item.get("company") in sample_companies}
            kept = []
            for fp in files:
                stem = fp.stem
                if stem in valid:
                    kept.append(fp)
                else:
                    parts = stem.rsplit("_page_", 1)
                    if len(parts) == 2 and parts[0] in valid:
                        kept.append(fp)
            logger.info(
                "MS staging: filtering by %d sample companies -> %d/%d files",
                len(sample_companies), len(kept), len(files),
            )
            files = kept

    staged = input_dir_for(corpus_tag)
    if staged.exists():
        shutil.rmtree(staged)
    staged.mkdir(parents=True)

    for fp in files:
        dest = staged / fp.name
        try:
            os.link(fp, dest)
        except OSError:
            shutil.copy2(fp, dest)

    logger.info("MS staging: %d files materialized at %s", len(files), staged)
    return staged


def _register_local_models_with_litellm() -> None:
    """LiteLLM rejects response_format/JSON-schema requests for unknown models.

    vLLM with Qwen3 actually supports structured output via guided_json, so
    we register our local model names with supports_response_schema=True.
    Without this, create_community_reports raises 'Model does not support
    response schemas' on every Leiden cluster.
    """
    import litellm

    base_meta = {
        "max_tokens": 16384,
        "max_input_tokens": 16384,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
        "litellm_provider": "openai",
        "supports_response_schema": True,
    }
    litellm.register_model({
        f"openai/{_GEN_MODEL_NAME}": {**base_meta, "mode": "chat"},
        f"openai/{_EMBED_MODEL_NAME}": {**base_meta, "mode": "embedding",
                                        "max_input_tokens": 8192, "output_vector_size": 1024},
    })


_ROUTER_INSTALLED = False


def _install_litellm_router_for_gen() -> None:
    """Monkey-patch litellm.acompletion to round-robin gen-chat across multiple
    vLLM endpoints (28000/GPU1 + 28010/GPU0). graphrag-llm calls bare
    `litellm.acompletion(**args)`; we intercept only when model matches our
    local gen model and delegate to a Router with simple-shuffle. Embedding +
    any other model passes through unchanged.
    """
    global _ROUTER_INSTALLED
    if _ROUTER_INSTALLED or len(_GEN_API_BASES) <= 1:
        return
    import contextvars
    import litellm
    from litellm import Router

    target = f"openai/{_GEN_MODEL_NAME}"
    model_list = [
        {
            "model_name": target,
            "litellm_params": {
                "model": target,
                "api_base": ab,
                "api_key": "EMPTY",
            },
        }
        for ab in _GEN_API_BASES
    ]
    router = Router(model_list=model_list, routing_strategy="simple-shuffle")

    # Capture the ORIGINAL acompletion before we replace litellm.acompletion.
    # Router internally calls `litellm.acompletion(...)`, which would re-enter
    # our wrapper and recurse forever. We use a contextvar to flag "we are
    # already inside Router" so the re-entry bypasses Router and uses the
    # original function — which is what Router actually expects to call.
    orig_acompletion = litellm.acompletion
    _in_router: contextvars.ContextVar[bool] = contextvars.ContextVar(
        "_ms_router_reentry", default=False,
    )

    async def _routed_acompletion(**kwargs):
        if _in_router.get():
            return await orig_acompletion(**kwargs)
        if kwargs.get("model") == target:
            kwargs.pop("api_base", None)
            token = _in_router.set(True)
            try:
                return await router.acompletion(**kwargs)
            finally:
                _in_router.reset(token)
        return await orig_acompletion(**kwargs)

    litellm.acompletion = _routed_acompletion
    _ROUTER_INSTALLED = True
    logger.info(
        "MS LiteLLM router installed for %s across %d endpoints: %s",
        target, len(_GEN_API_BASES), _GEN_API_BASES,
    )


def build_config(corpus_tag: str, staged_input_dir: Path):
    """Construct a GraphRagConfig pointing LiteLLM at our local vLLM."""
    _register_local_models_with_litellm()
    _install_litellm_router_for_gen()

    from graphrag.config.models.graph_rag_config import GraphRagConfig
    from graphrag_cache import CacheConfig
    from graphrag_input import InputConfig
    from graphrag_llm.config.model_config import ModelConfig
    from graphrag_storage import StorageConfig, StorageType
    from graphrag_vectors import IndexSchema, VectorStoreConfig

    out_dir = output_dir_for(corpus_tag)
    cache_dir = cache_dir_for(corpus_tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # vLLM 4B quirks:
    # - encoding_format="float" required (LiteLLM 1.83 sends None which vLLM 0.15 rejects).
    # - max_tokens: keep modest so a runaway entity-extraction doesn't blow chunk context.
    # - extra_body.guided_json supported by vLLM but not configured here; rely on json_repair fallback.
    completion_call_args = {
        "temperature": 0.0,
        "max_tokens": 1500,
    }

    cfg = GraphRagConfig(
        completion_models={
            "default_completion_model": ModelConfig(
                type="litellm",
                model_provider="openai",
                model=_GEN_MODEL_NAME,
                api_base=_GEN_API_BASE,
                api_key="EMPTY",
                call_args=completion_call_args,
            ),
        },
        embedding_models={
            "default_embedding_model": ModelConfig(
                type="litellm",
                model_provider="openai",
                model=_EMBED_MODEL_NAME,
                api_base=_EMBED_API_BASE,
                api_key="EMPTY",
                call_args={"encoding_format": "float"},
            ),
        },
        input=InputConfig(file_pattern=r".*\.txt$"),
        input_storage=StorageConfig(
            type=StorageType.File,
            base_dir=str(staged_input_dir),
        ),
        output_storage=StorageConfig(
            type=StorageType.File,
            base_dir=str(out_dir),
        ),
        cache=CacheConfig(
            storage=StorageConfig(type=StorageType.File, base_dir=str(cache_dir)),
        ),
        # Qwen3-Embedding-0.6B emits 1024-dim vectors; default IndexSchema
        # assumes 3072 (text-embedding-3-large). Without the override, lancedb
        # rejects the embedding parquet on a FixedSizeList shape mismatch.
        # Keys MUST match generate_text_embeddings.py's embedded_fields:
        # entity_description / community_full_content / text_unit_text
        # (graphrag.config.embeddings constants), not arbitrary names.
        vector_store=VectorStoreConfig(
            db_uri=str(out_dir / "lancedb"),
            index_schema={
                name: IndexSchema(index_name=name, vector_size=1024)
                for name in (
                    "entity_description",
                    "community_full_content",
                    "text_unit_text",
                )
            },
        ),
    )
    # MS pipeline gates extract_graph + summarize via asyncio.Semaphore(num_threads=concurrent_requests).
    # vLLM 4B handles 30+ parallel reqs comfortably (peak observed: 14 running + 7 waiting at limit 16
    # → fully saturated). Bump to 48 to drive the queue and shave wall-clock on the 33k-text_unit corpus.
    cfg.concurrent_requests = int(os.environ.get("RAG_MS_CONCURRENT_REQUESTS", "48"))
    return cfg


async def run_official_index(
    dataset_path: str,
    corpus_tag: str,
    sample_companies: Optional[list[str]] = None,
) -> None:
    """Stage inputs, build config, run the standard MS pipeline."""
    from graphrag.api.index import build_index
    from graphrag.config.enums import IndexingMethod

    staged_input = _stage_input_files(dataset_path, corpus_tag, sample_companies)

    config = build_config(corpus_tag, staged_input)
    out_dir = output_dir_for(corpus_tag)

    logger.info(
        "MS official indexing: corpus_tag=%s, %d input files, output=%s, "
        "gen=%s embed=%s",
        corpus_tag, len(list(staged_input.iterdir())), out_dir,
        _GEN_API_BASE, _EMBED_API_BASE,
    )

    results = await build_index(
        config=config,
        method=IndexingMethod.Standard,
        verbose=False,
    )

    failures = [r for r in results if getattr(r, "errors", None)]
    if failures:
        logger.warning("MS pipeline produced %d workflow(s) with errors", len(failures))
        for r in failures:
            logger.warning("  workflow=%s errors=%s", getattr(r, "workflow", "?"), r.errors)

    # Sanity: verify expected parquet artifacts.
    expected = ["entities.parquet", "relationships.parquet", "communities.parquet",
                "community_reports.parquet", "text_units.parquet", "documents.parquet"]
    missing = [name for name in expected if not (out_dir / name).exists()]
    if missing:
        logger.warning("MS pipeline missing expected artifacts: %s", missing)
    else:
        logger.info("MS pipeline produced all expected parquet files at %s", out_dir)
