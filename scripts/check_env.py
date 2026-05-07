import asyncio
import os
import httpx
from neo4j import AsyncGraphDatabase
import sys

# Ensure project root is on sys.path so `from core.* import` works regardless
# of the cwd this script is invoked from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.vllm_client import VLLMClient
from core.neo4j_service import Neo4jService

async def check_vllm(name, url):
    api_key = os.environ.get("VLLM_API_KEY", "EMPTY")
    print(f"Checking {name} at {url}...", end=" ", flush=True)
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        async with httpx.AsyncClient(timeout=10.0, headers=headers) as client:
            # Reranker service uses /health instead of /v1/models
            if "rerank" in name.lower():
                # Extract base URL (remove /v1 if present)
                base_url = url.split("/v1")[0].rstrip("/")
                check_url = f"{base_url}/health"
            else:
                check_url = f"{url.rstrip('/')}/models"
                
            resp = await client.get(check_url)
            if resp.status_code == 200:
                print("✅ OK")
                return True
            else:
                print(f"⚠️ Status {resp.status_code} at {check_url}")
                return False
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

async def check_neo4j():
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    pwd = os.environ.get("NEO4J_PASSWORD", "1q2w3e4r")
    print(f"Checking Neo4j at {uri}...", end=" ", flush=True)
    try:
        driver = AsyncGraphDatabase.driver(uri, auth=(user, pwd))
        async with driver.session() as session:
            await session.run("RETURN 1")
        await driver.close()
        print("✅ OK")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

async def main():
    print("=== HypoReflect Environment Connectivity Check (uv-based) ===\n")
    print(f"Python Interpreter: {sys.executable}")
    print(f"Working Directory: {os.getcwd()}\n")
    
    vllm = VLLMClient()
    # Use environment variables from vllm.env if needed, but here we use what's in core/config.py
    results = await asyncio.gather(
        check_vllm("Generation Service", vllm.vllm_url),
        check_vllm("Embedding Service", vllm.embed_url),
        check_vllm("OCR Service", vllm.ocr_url),
        check_vllm("Reranker Service", os.environ.get("VLLM_RERANK_URL", "http://localhost:18083/v1")),
        check_neo4j()
    )
    
    if all(results):
        print("\n🚀 All systems are ready for benchmark (Local UV Environment)!")
    else:
        print("\n⚠️ Some services are unreachable. Please ensure vLLM and Neo4j are running locally.")

if __name__ == "__main__":
    asyncio.run(main())