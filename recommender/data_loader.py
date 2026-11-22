import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def load_products():
    """Загрузка каталога бытовой техники."""

    path = DATA_DIR / "products.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Файл каталога не найден: {path}"
        )

    products = pd.read_csv(path)

    required_columns = {
        "id", "name", "category",
        "brand", "price", "description", "features"
    }

    if not required_columns.issubset(products.columns):
        raise ValueError("Некорректная структура products.csv")

    products["price"] = pd.to_numeric(
        products["price"], errors="coerce"
    )

    products = products.dropna(subset=["id", "price"])

    products["id"] = products["id"].astype(str)

    return products


def load_ratings(path=None):
    """Загрузка оценок пользователей."""

    path = Path(path) if path is not None else DATA_DIR / "ratings.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Файл оценок не найден: {path}"
        )

    ratings = pd.read_csv(path)

    required_columns = {
        "user_id", "product_id", "rating"
    }

    if not required_columns.issubset(ratings.columns):
        raise ValueError("Некорректная структура ratings.csv")

    ratings["rating"] = pd.to_numeric(
        ratings["rating"], errors="coerce"
    )

    ratings = ratings.dropna(
        subset=["user_id", "product_id", "rating"]
    )

    ratings["user_id"] = ratings["user_id"].astype(str)
    ratings["product_id"] = ratings["product_id"].astype(str)

    ratings = ratings[
        ratings["rating"].between(1, 5)
    ]

    # Если пользователь несколько раз оценил товар,
    # используем среднее значение оценок.
    ratings = ratings.groupby(
        ["user_id", "product_id"],
        as_index=False
    )["rating"].mean()

    return ratings
