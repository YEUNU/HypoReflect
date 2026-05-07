import asyncio
import httpx

async def probe(port):
    urls = [
        f"http://localhost:{port}/v1/models",
        f"http://localhost:{port}/health",
        f"http://localhost:{port}/"
    ]
    print(f"Probing port {port}...")
    async with httpx.AsyncClient(timeout=2.0) as client:
        for url in urls:
            try:
                resp = await client.get(url)
                print(f"  {url} -> {resp.status_code}")
                if resp.status_code == 200:
                    try:
                        print(f"    Body: {resp.text[:100]}")
                    except:
                        pass
            except Exception as e:
                print(f"  {url} -> Error: {e}")

async def main():
    ports = [8000, 8001, 8002, 8003, 18082, 18083, 28000, 28001]
    await asyncio.gather(*(probe(p) for p in ports))

if __name__ == "__main__":
    asyncio.run(main())
