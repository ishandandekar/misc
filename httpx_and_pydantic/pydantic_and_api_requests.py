import httpx
import pydantic
from pprint import pprint


class Variant(pydantic.BaseModel):
    title: str
    sku: str
    price: str


class Product(pydantic.BaseModel):
    id: int
    title: str
    variants: list[Variant]


def fetch_data():
    resp = httpx.get("https://www.allbirds.co.uk/products.json")
    return resp.json()["products"]


def main():
    products = fetch_data()
    for product in products:
        item = Product(**product)
        pprint(item.model_dump())
        break


if __name__ == "__main__":
    main()
