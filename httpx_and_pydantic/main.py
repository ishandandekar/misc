from typing import Any
import time
import httpx
import asyncio

BASE_URL = "https://httpbin.org"


async def fetch_get(client: httpx.AsyncClient) -> Any:
    response = await client.get(f"{BASE_URL}/get")
    return response.json()


async def fetch_post(client: httpx.AsyncClient) -> Any:
    data_to_post = {"key": "value"}
    response = await client.post(f"{BASE_URL}/post", json=data_to_post)
    return response.json()


async def fetch_put(client: httpx.AsyncClient) -> Any:
    data_to_put = {"key": "updated_value"}
    response = await client.put(f"{BASE_URL}/put", json=data_to_put)
    return response.json()


async def main() -> None:
    start = time.perf_counter()
    async with httpx.AsyncClient() as client:
        tasks = [
            fetch_get(client=client),
            fetch_post(client=client),
            fetch_put(client=client),
        ]
        results = await asyncio.gather(*tasks)

    # for result in results:
    # print(result)
    print(f"Time take: {time.perf_counter() - start:.2f} seconds.")
    return results


if __name__ == "__main__":
    r = asyncio.run(main())
    print(r)
