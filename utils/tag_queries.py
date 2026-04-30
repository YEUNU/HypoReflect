import json
import asyncio
import os
import logging
from typing import List, Dict, Any
from core.vllm_client import get_llm_client
from utils.prompts import QUERY_CATEGORIZATION_PROMPT
from core.config import RAGConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryTagger:
    def __init__(self, model_id: str = "local"):
        self.llm = get_llm_client(model_id)
        self.semaphore = asyncio.Semaphore(5) # limit concurrency for tagging

    async def tag_query(self, item: Dict[str, Any]) -> Dict[str, Any]:
        with open("tag_queries.log", "a") as logf:
            logf.write(f"Start tagging item {item.get('_id')}\n")
        
        query = item.get("query")
        evidence = item.get("evidence_text", "")
        
        prompt = QUERY_CATEGORIZATION_PROMPT.format(query=query, evidence=evidence[:2000]) # truncated evidence
        
        if not query:
            with open("tag_queries.log", "a") as logf:
                logf.write(f"Skipping empty query item: {item.get('_id', 'unknown')}\n")
            return item

        async with self.semaphore:
            try:
                with open("tag_queries.log", "a") as logf:
                    logf.write(f"Tagging query: {query[:30]}...\n")
                
                response = await self.llm.generate_response([{"role": "user", "content": prompt}], temperature=0.0)
                
                with open("tag_queries.log", "a") as logf:
                    logf.write(f"Response received for: {query[:30]}...\n")
                
                # Parse CATEGORY and REASON
                category = "Unknown"
                reason = ""
                for line in response.split("\n"):
                    if line.startswith("CATEGORY:"):
                        category = line.replace("CATEGORY:", "").strip().lower().capitalize()
                    elif line.startswith("REASON:"):
                        reason = line.replace("REASON:", "").strip()
                
                item["category"] = category
                item["category_reason"] = reason
                return item
            except Exception as e:
                with open("tag_queries.log", "a") as logf:
                    logf.write(f"Error tagging query {item.get('_id')}: {e}\n")
                logger.error(f"Error tagging query {item.get('_id')}: {e}")
                item["category"] = "Error"
                return item

    async def run(self, input_file: str, output_file: str, limit: int = None):
        with open("tag_queries.log", "w") as logf:
            logf.write("Starting run...\n")
            
        if not os.path.exists(input_file):
            with open("tag_queries.log", "a") as logf:
                logf.write(f"Input file not found: {input_file}\n")
            return

        with open(input_file, 'r') as f:
            data = json.load(f)
        
        if limit:
            data = data[:limit]
            
        with open("tag_queries.log", "a") as logf:
            logf.write(f"Tagging {len(data)} queries from {input_file}...\n")
        
        tasks = [self.tag_query(item) for item in data]
        results = await asyncio.gather(*tasks)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            
        with open("tag_queries.log", "a") as logf:
            logf.write(f"Saved tagged queries to {output_file}\n")

if __name__ == "__main__":
    import argparse
    from main import get_sample_companies
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true", help="Tag only queries from sample companies")
    parser.add_argument("--limit", type=int, help="Limit number of queries to tag")
    args = parser.parse_args()
    
    tagger = QueryTagger()
    input_path = os.path.join(RAGConfig.PROJECT_ROOT, "data/financebench_queries.json")
    
    if args.sample:
        output_path = os.path.join(RAGConfig.PROJECT_ROOT, "data/financebench_queries_sample_tagged.json")
        companies = get_sample_companies()
        
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        filtered_data = [item for item in data if item.get("company") in companies]
        logger.info(f"Sample mode: Filtered {len(data)} -> {len(filtered_data)} queries for companies: {companies}")
        
        # Save temp filtered file for the tagger
        temp_input = os.path.join(RAGConfig.PROJECT_ROOT, "data/temp_sample_queries.json")
        with open(temp_input, 'w') as f:
            json.dump(filtered_data, f)
        
        asyncio.run(tagger.run(temp_input, output_path, limit=args.limit))
        os.remove(temp_input)
    else:
        output_path = os.path.join(RAGConfig.PROJECT_ROOT, "data/financebench_queries_tagged.json")
        asyncio.run(tagger.run(input_path, output_path, limit=args.limit))
