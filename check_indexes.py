import asyncio
from core.neo4j_service import Neo4jService

async def check():
    n = Neo4jService()
    
    print("=== Indexes ===")
    indexes = await n.execute_query('SHOW INDEXES')
    for idx in indexes:
        print(f"Index: {idx['name']} | Type: {idx['type']} | State: {idx['state']}")
    
    print("\n=== Graph Statistics ===")
    counts = await n.execute_query("""
        MATCH (c) RETURN labels(c)[0] as label, count(*) as count
    """)
    for row in counts:
        print(f"Nodes ({row['label']}): {row['count']}")
        
    rels = await n.execute_query("""
        MATCH ()-[r]->() RETURN type(r) as type, count(*) as count
    """)
    for row in rels:
        print(f"Relationships ({row['type']}): {row['count']}")
    
    await n.global_close()

if __name__ == "__main__":
    asyncio.run(check())
