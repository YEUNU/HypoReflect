import asyncio
from neo4j import AsyncGraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

async def test_neo4j():
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")
    
    print(f"Connecting to {uri} as {user}...")
    try:
        driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        async with driver:
            await driver.verify_connectivity()
            print("✅ Successfully connected to Neo4j!")
    except Exception as e:
        print(f"❌ Failed to connect to Neo4j: {e}")

if __name__ == "__main__":
    asyncio.run(test_neo4j())
