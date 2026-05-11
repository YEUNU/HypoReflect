"""Official HopRAG indexing wired to local vLLM + our Neo4j namespacing.

`third_party/HopRAG` is the upstream package. It's not pip-installable: it
imports `config` and `tool` as top-level modules and bakes config-time vars
(edge_name, embed_dim, deployment_sign, cypher templates) into module
constants. We:

1. Prepend the package dir to sys.path.
2. Import `config`, override its attributes for local vLLM + corpus-tagged
   labels, and recompile the cypher templates that string-concatenated
   `edge_name` at module load.
3. Monkey-patch `tool.load_embed_model` / `tool.get_doc_embeds` to use our
   vLLM embedding endpoint (avoids loading SentenceTransformer locally).
4. Then `import HopBuilder`, which picks up the patched config.

`HopBuilder.create_edge` does pairwise question similarity, which is O(N²) per
group. We group documents by company (FinanceBench is company-anchored, like
hyporeflect §3.1.4 same-company HOP filter) so each group stays tractable.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("HypoReflect")

_HOPRAG_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "HopRAG"
_GEN_API_BASE = os.environ.get("RAG_HOP_GEN_API_BASE", "http://localhost:28000/v1")
_GEN_MODEL_NAME = os.environ.get("VLLM_SERVED_MODEL_NAME", "generation-model")
_EMBED_API_BASE = os.environ.get("RAG_HOP_EMBED_API_BASE", "http://localhost:18082/v1")
_EMBED_MODEL_NAME = os.environ.get("RAG_HOP_EMBED_MODEL_NAME", "embedding-model")
_EMBED_DIM = int(os.environ.get("RAG_HOP_EMBED_DIM", "1024"))

_OUTPUT_ROOT = Path(os.environ.get("RAG_HOP_OUTPUT_ROOT", "data/hoprag_output"))


def output_dir_for(corpus_tag: str) -> Path:
    return (_OUTPUT_ROOT / corpus_tag).resolve()


def cache_dir_for(corpus_tag: str) -> Path:
    return (_OUTPUT_ROOT / corpus_tag / "_cache").resolve()


def input_dir_for(corpus_tag: str) -> Path:
    return (_OUTPUT_ROOT / corpus_tag / "_input").resolve()


# ---------------------------------------------------------------- file staging

def _stage_input_files(
    dataset_path: str,
    corpus_tag: str,
    sample_companies: Optional[list[str]],
) -> tuple[Path, dict[str, str]]:
    """Materialize a tag-scoped input dir, return (dir, doc_to_company map)."""
    src_root = Path(dataset_path)
    if not src_root.exists():
        raise FileNotFoundError(f"dataset_path not found: {dataset_path}")

    files = sorted(p for p in src_root.iterdir() if p.suffix in (".txt", ".md"))

    doc_info_path = Path("data/financebench_document_information.jsonl")
    doc_to_company: dict[str, str] = {}
    if doc_info_path.exists():
        with doc_info_path.open() as fh:
            for line in fh:
                rec = json.loads(line)
                if rec.get("doc_name") and rec.get("company"):
                    doc_to_company[rec["doc_name"]] = rec["company"]

    if sample_companies:
        valid = {n for n, c in doc_to_company.items() if c in sample_companies}
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
            "HopRAG staging: filtering by %d sample companies -> %d/%d files",
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

    logger.info("HopRAG staging: %d files materialized at %s", len(files), staged)

    # filename → company map for grouped edge construction
    file_to_company: dict[str, str] = {}
    for fp in files:
        stem = fp.stem
        if stem in doc_to_company:
            file_to_company[fp.name] = doc_to_company[stem]
        else:
            parts = stem.rsplit("_page_", 1)
            if len(parts) == 2 and parts[0] in doc_to_company:
                file_to_company[fp.name] = doc_to_company[parts[0]]
            else:
                file_to_company[fp.name] = "_unknown"
    return staged, file_to_company


# ---------------------------------------------------------------- monkey patch

class _VLLMEmbedClient:
    """Drop-in replacement for SentenceTransformer.encode().

    Calls our vLLM /v1/embeddings (same backing model as the rest of the stack
    so HopRAG nodes/edges live in the same embedding space as hyporeflect's,
    which keeps the architectural comparison apples-to-apples).
    """

    def __init__(self, base_url: str, model: str, dim: int):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dim = dim
        import requests
        self._sess = requests.Session()

    def encode(self, documents, normalize_embeddings: bool = True, device=None):
        _ = normalize_embeddings, device
        if documents is None:
            return np.zeros((0, self.dim), dtype=np.float32)
        if isinstance(documents, str):
            single = True
            documents = [documents]
        else:
            single = False

        if not documents:
            return np.zeros((0, self.dim), dtype=np.float32)

        # vLLM batches via continuous batching; chunk to keep payloads sane.
        chunk = 64
        out = []
        for i in range(0, len(documents), chunk):
            batch = documents[i : i + chunk]
            r = self._sess.post(
                f"{self.base_url}/embeddings",
                json={
                    "model": self.model,
                    "input": batch,
                    "encoding_format": "float",
                },
                headers={"Authorization": "Bearer EMPTY"},
                timeout=180,
            )
            r.raise_for_status()
            data = r.json()["data"]
            for d in data:
                out.append(d["embedding"])

        arr = np.asarray(out, dtype=np.float32)
        return arr[0] if single else arr


def _install_optional_stubs() -> None:
    """Stub HopRAG's heavy/Chinese-NLP deps that we don't need (paddlenlp +
    sentence_transformers + modelscope). They get imported at module load by
    third_party/HopRAG/tool.py but we replace their downstream calls."""
    import types
    if "paddlenlp" not in sys.modules:
        m = types.ModuleType("paddlenlp")
        m.Taskflow = lambda *a, **kw: (lambda _t: [])
        sys.modules["paddlenlp"] = m
    if "sentence_transformers" not in sys.modules:
        m = types.ModuleType("sentence_transformers")
        class _ST:
            def __init__(self, *a, **kw): pass
            def encode(self, docs, **kw):
                return [0.0] * 768 if isinstance(docs, str) else [[0.0] * 768 for _ in docs]
        m.SentenceTransformer = _ST
        sys.modules["sentence_transformers"] = m
    if "modelscope" not in sys.modules:
        m = types.ModuleType("modelscope")
        class _Dummy:
            @classmethod
            def from_pretrained(cls, *a, **kw): return cls()
            def eval(self): return self
            def to(self, *a, **kw): return self
            @property
            def device(self): return "cpu"
            def __call__(self, *a, **kw):
                return types.SimpleNamespace(logits=[0.0])
        m.AutoModelForCausalLM = _Dummy
        m.AutoModelForSequenceClassification = _Dummy
        m.AutoTokenizer = _Dummy
        sys.modules["modelscope"] = m


def _setup_hoprag_modules(corpus_tag: str) -> None:
    """Prep sys.path + override config + tool BEFORE HopBuilder imports."""
    _install_optional_stubs()
    if str(_HOPRAG_ROOT) not in sys.path:
        sys.path.insert(0, str(_HOPRAG_ROOT))

    import config

    config.local_base = _GEN_API_BASE
    config.local_key = "EMPTY"
    config.local_model_name = _GEN_MODEL_NAME
    config.query_generator_model = _GEN_MODEL_NAME
    config.traversal_model = _GEN_MODEL_NAME
    config.default_gpt_model = _GEN_MODEL_NAME

    config.embed_model = "qwen3_embed_via_vllm"
    config.embed_model_dict = {"qwen3_embed_via_vllm": "(vllm-served)"}
    config.embed_dim = _EMBED_DIM
    config.signal = "\n\n"
    config.max_thread_num = max(1, int(os.environ.get("RAG_HOP_MAX_THREADS", "8")))

    safe = "".join(c if c.isalnum() else "_" for c in corpus_tag)
    config.dataset_name = "financebench"
    config.node_name = f"HO_{safe}"
    config.edge_name = f"HO_{safe}_p2a"
    config.generator_label = f"HO_{safe}_"
    config.node_dense_index_name = f"HO_{safe}_node_dense_idx"
    config.edge_dense_index_name = f"HO_{safe}_edge_dense_idx"
    config.node_sparse_index_name = f"HO_{safe}_node_sparse_idx"
    config.edge_sparse_index_name = f"HO_{safe}_edge_sparse_idx"

    # Cypher templates were string-concat'd with the OLD edge_name at module
    # load. Rebuild with the new one.
    config.create_pending2answerable = (
        "MATCH (a), (b) WHERE id(a) = $id1 AND id(b) = $id2 "
        f"CREATE (a)-[r:{config.edge_name} "
        "{keywords: $keywords, embed: $embed, question: $answerable_question}]->(b)"
    )
    config.create_abstract2answerable = (
        "MATCH (a), (b) WHERE id(a) = $abstract_id AND id(b) = $id2 "
        f"CREATE (a)-[r:{config.edge_name} "
        "{keywords: $keywords, embed: $embed, question: $answerable_question}]->(b)"
    )

    # Reflect new local_model_name in deployment_sign so _get_chat_completion
    # routes to our vLLM rather than gpt.
    config.deployment_sign = {
        "gpt": {
            "base": getattr(config, "gpt_base", ""),
            "key": getattr(config, "gpt_key", ""),
            "default_model": "gpt-4o-mini",
        },
        config.local_model_name: {"base": config.local_base, "key": config.local_key},
    }

    # Neo4j connection from our env (matches Neo4jService defaults).
    config.neo4j_url = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    config.neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
    config.neo4j_password = os.environ.get("NEO4J_PASSWORD", "1q2w3e4r")
    config.neo4j_dbname = os.environ.get("NEO4J_DATABASE", "neo4j")

    # Patch tool to use vLLM embeddings instead of local SentenceTransformer.
    import tool
    embed_client = _VLLMEmbedClient(_EMBED_API_BASE, _EMBED_MODEL_NAME, _EMBED_DIM)
    tool.load_embed_model = lambda _name: embed_client

    # Default get_doc_embeds calls model.encode(...).tolist() — our wrapper
    # already returns numpy with .tolist(), so the original works as-is.

    # Replace paddlenlp-based POS tagging with spaCy. Original keeps content
    # words (nouns/proper-nouns/verbs/adj) and drops function words. Without
    # this, get_ner_eng falls back to character-level splitting (we saw
    # keywords like [' ', '0', '2', 'B'] in the smoke), which trashes
    # sparse_similarity in HopBuilder.create_edge.
    tool.get_ner_eng = _spacy_ner_eng

    _patch_hopbuilder_for_pandas2()


_SPACY_NLP = None
_KEEP_POS = {"NOUN", "PROPN", "VERB", "ADJ", "NUM"}


def _spacy_ner_eng(text: str):
    """spaCy substitute for paddlenlp Taskflow('pos_tagging').

    Returns a list of unique content-word lemmas. Filters punctuation, stop
    words, and function-word POS tags. Result feeds into node 'keywords'
    sets used by sparse_similarity in HopBuilder.create_edge.
    """
    global _SPACY_NLP
    if _SPACY_NLP is None:
        import spacy
        _SPACY_NLP = spacy.load("en_core_web_sm", disable=["parser"])
    doc = _SPACY_NLP(str(text or ""))
    seen = set()
    out = []
    for tok in doc:
        if tok.is_punct or tok.is_space or tok.is_stop:
            continue
        if tok.pos_ not in _KEEP_POS:
            continue
        lemma = tok.lemma_.lower().strip()
        if not lemma or len(lemma) < 2:
            continue
        if lemma in seen:
            continue
        seen.add(lemma)
        out.append(lemma)
    return out


def _patch_hopbuilder_for_pandas2() -> None:
    """HopBuilder.create_edge does
        df.apply(lambda x: x['kw_x'].union(x['kw_y']), axis=1)
    which returns a Series of set objects. pandas 2.x expands those into
    multiple columns when assigned back, raising
    'Cannot set a DataFrame with multiple columns to the single column'.
    Wrap the union result in a list so pandas treats it as a scalar."""
    import HopBuilder

    if getattr(HopBuilder.QABuilder.create_edge, "_patched_for_pandas2", False):
        return

    import inspect, textwrap, pandas as pd

    src = inspect.getsource(HopBuilder.QABuilder.create_edge)
    src = textwrap.dedent(src)

    # pandas 2.x expands lambda-returned set/list into multiple columns when
    # assigned. Rewrite both `apply(...)` lines to use list comprehensions,
    # which always yield a single Series of scalars.
    replacements = [
        (
            "cartesian1['keywords_both']=cartesian1.apply(lambda x:x['keywords_x'].union(x['keywords_y']),axis=1)",
            "cartesian1['keywords_both']=[set(kx).union(set(ky)) for kx,ky in zip(cartesian1['keywords_x'],cartesian1['keywords_y'])]",
        ),
        (
            "cartesian2['keywords_both']=cartesian2.apply(lambda x:x['keywords_x'].union(x['keywords_y']),axis=1)",
            "cartesian2['keywords_both']=[set(kx).union(set(ky)) for kx,ky in zip(cartesian2['keywords_x'],cartesian2['keywords_y'])]",
        ),
    ]
    for old, new in replacements:
        if old not in src:
            raise RuntimeError(
                f"HopBuilder.create_edge source pattern not found: {old[:60]}..."
            )
        src = src.replace(old, new)

    # Bind into the HopBuilder module namespace so cypher templates etc resolve.
    namespace = dict(HopBuilder.__dict__)
    namespace.update({"pd": pd})
    exec(compile(src, "<hop_create_edge_patched>", "exec"), namespace)
    patched = namespace["create_edge"]
    patched._patched_for_pandas2 = True  # type: ignore[attr-defined]
    HopBuilder.QABuilder.create_edge = patched
    logger.info("HopRAG: patched QABuilder.create_edge for pandas-2 compatibility")


# ---------------------------------------------------------------- driver

def _group_files_by_company(file_to_company: dict[str, str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for fname, company in file_to_company.items():
        groups.setdefault(company, []).append(fname)
    return groups


def _run_official_index_blocking(
    dataset_path: str,
    corpus_tag: str,
    sample_companies: Optional[list[str]],
) -> None:
    """Synchronous driver — HopBuilder is sync, so we call it directly and
    let the orchestrator wrap us in run_in_executor."""
    _setup_hoprag_modules(corpus_tag)

    # Now safe to import HopBuilder (it does `from config import *`).
    import HopBuilder
    import config

    staged_input, file_to_company = _stage_input_files(
        dataset_path, corpus_tag, sample_companies
    )
    cache_dir = cache_dir_for(corpus_tag)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: build nodes (LLM-heavy: title/keywords + answerable+pending Qs).
    # main_nodes auto-resumes via docid2nodes.json cache.
    logger.info(
        "HopRAG official indexing: corpus_tag=%s, %d input files, output=%s, "
        "node_name=%s edge_name=%s gen=%s embed=%s",
        corpus_tag, len(file_to_company), output_dir_for(corpus_tag),
        config.node_name, config.edge_name, _GEN_API_BASE, _EMBED_API_BASE,
    )

    HopBuilder.main_nodes(
        cache_dir=str(cache_dir),
        docs_dir=str(staged_input),
        label=config.node_name,
        start_index=0,
        span=len(file_to_company) + 100,
        offline=True,
    )

    # Stage 2: insert nodes into Neo4j (assigning real node IDs) + edges +
    # indices, grouped by company (FinanceBench is company-anchored — cross-
    # company edges add noise per CLAUDE.md hyporeflect §3.1.4).
    docid2nodes_path = cache_dir / "docid2nodes.json"
    node2q_path = cache_dir / "node2questiondict.pkl"
    if not (docid2nodes_path.exists() and node2q_path.exists()):
        logger.error("HopRAG: stage-1 cache missing, cannot build edges")
        return

    builder = HopBuilder.QABuilder(done=set(), label=config.node_name)

    # create_nodes_cache reads cache pickle (which has (node_dict, q_dict)
    # tuples) and writes nodes to Neo4j, returning the dict-only form
    # (new_node2questiondict) that create_edge expects.
    new_docid2nodes, new_node2q = builder.create_nodes_cache(str(cache_dir))
    logger.info(
        "HopRAG: inserted %d nodes across %d docs into Neo4j (label=%s)",
        sum(len(v) for v in new_docid2nodes.values()), len(new_docid2nodes),
        config.node_name,
    )

    # HopRAG-native schema (text/keywords/embed) drops doc provenance, which
    # breaks the bench's doc_match metric. Backfill `source` (filename stem
    # = financebench doc_name) and `company` per node so retrieval results
    # carry the same provenance as hyporeflect/ms_graphrag baselines.
    if builder.driver is None:
        from neo4j import GraphDatabase
        builder.driver = GraphDatabase.driver(
            config.neo4j_url,
            auth=(config.neo4j_user, config.neo4j_password),
            database=config.neo4j_dbname,
        )
    backfill_pairs = []
    for doc_id, node_ids in new_docid2nodes.items():
        stem = Path(doc_id).stem
        company = file_to_company.get(doc_id, "_unknown")
        for nid in node_ids:
            backfill_pairs.append({"id": int(nid), "source": stem, "company": company})
    if backfill_pairs:
        with builder.driver.session() as s:
            s.run(
                "UNWIND $rows AS row MATCH (n) WHERE id(n) = row.id "
                "SET n.source = row.source, n.company = row.company",
                {"rows": backfill_pairs},
            )
        logger.info("HopRAG: backfilled source/company on %d nodes", len(backfill_pairs))

    groups = _group_files_by_company(file_to_company)
    logger.info("HopRAG: building edges per-company over %d groups", len(groups))

    for company, doc_list in groups.items():
        docs_in_group = [d for d in doc_list if d in new_docid2nodes]
        if not docs_in_group:
            continue
        sub_docid2nodes = {d: new_docid2nodes[d] for d in docs_in_group}
        sub_node2q = {
            (nid, did): new_node2q[(nid, did)]
            for did in docs_in_group
            for nid in new_docid2nodes[did]
            if (nid, did) in new_node2q
        }
        if not sub_node2q:
            continue
        try:
            builder.create_edge(sub_node2q, sub_docid2nodes)
            logger.info("  edges built for company=%s (%d docs)", company, len(docs_in_group))
        except Exception as e:
            logger.warning("  edge build failed for company=%s: %s", company, e)

    # Stage 3: vector + fulltext indices.
    builder.create_index()
    if builder.driver is not None:
        builder.driver.close()
        builder.driver = None
    logger.info("HopRAG official indexing complete for %s", corpus_tag)


async def run_official_index(
    dataset_path: str,
    corpus_tag: str,
    sample_companies: Optional[list[str]] = None,
) -> None:
    """Async wrapper around the sync HopBuilder driver."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        _run_official_index_blocking,
        dataset_path,
        corpus_tag,
        sample_companies,
    )
