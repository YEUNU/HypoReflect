import os
from neo4j import AsyncGraphDatabase
from typing import List, Dict, Any, Optional
import re
import logging

class Neo4jService:
    _driver = None

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Local connection info
        self.uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        self.user = os.environ.get("NEO4J_USER", "neo4j")
        self.password = os.environ.get("NEO4J_PASSWORD", "1q2w3e4r")
        
        if Neo4jService._driver is None:
            Neo4jService._driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                keep_alive=True,
                max_connection_pool_size=int(os.environ.get("NEO4J_POOL_SIZE", "50")),
                connection_acquisition_timeout=30,
            )
        self.driver = Neo4jService._driver

    async def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        async with self.driver.session() as session:
            result = await session.run(query, parameters or {})  # type: ignore
            return [dict(record) async for record in result]

    async def close(self):
        await Neo4jService.global_close()

    @classmethod
    async def global_close(cls):
        if cls._driver:
            await cls._driver.close()
            cls._driver = None
