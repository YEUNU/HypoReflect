import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.neo4j_service import Neo4jService

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("reset-neo4j")

async def reset():
    neo4j = Neo4jService()
    
    logger.info("Resetting Neo4j database (Deleting all data and indexes)...")
    try:
        # 1. Drop all constraints
        try:
            constraints = await neo4j.execute_query("SHOW CONSTRAINTS")
            for c in constraints:
                name = c['name']
                logger.info(f"Dropping constraint: {name}")
                await neo4j.execute_query(f"DROP CONSTRAINT {name}")
        except Exception as e:
            logger.warning(f"Could not drop constraints: {e}")

        # 2. Delete all relationships in batches
        logger.info("Deleting all relationships in batches...")
        total_rels = 0
        while True:
            res = await neo4j.execute_query("MATCH ()-[r]->() WITH r LIMIT 10000 DELETE r RETURN count(*) as count")
            count = res[0]['count']
            total_rels += count
            if count == 0:
                break
            logger.info(f"Deleted {count} relationships (Total: {total_rels})...")
        
        # 3. Delete all nodes in batches
        logger.info("Deleting all nodes in batches...")
        total_nodes = 0
        while True:
            res = await neo4j.execute_query("MATCH (n) WITH n LIMIT 10000 DETACH DELETE n RETURN count(*) as count")
            count = res[0]['count']
            total_nodes += count
            if count == 0:
                break
            logger.info(f"Deleted {count} nodes (Total: {total_nodes})...")
        
        # 4. Drop all indexes
        indexes = await neo4j.execute_query("SHOW INDEXES")
        for idx in indexes:
            # Skip internal/system indexes if necessary (lookup indexes)
            if idx['type'] in ['LOOKUP', 'FULLTEXT', 'VECTOR', 'RANGE']:
                try:
                    await neo4j.execute_query(f"DROP INDEX {idx['name']}")
                    logger.info(f"Dropped index: {idx['name']}")
                except Exception as e:
                    logger.warning(f"Could not drop index {idx['name']}: {e}")
        
        logger.info("Database reset complete.")
    except Exception as e:
        logger.error(f"Error during reset: {e}")
    finally:
        await neo4j.global_close()

if __name__ == "__main__":
    asyncio.run(reset())
