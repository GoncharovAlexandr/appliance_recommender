"""Создать искусственные оценки для эксперимента, не затрагивая данные приложения."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from recommender.data_loader import load_products


def generate(products, users=200, seed=2026):
    if users < 1 or len(products) < 3 or not products["id"].is_unique:
        raise ValueError("Нужен хотя бы один пользователь и три товара с уникальными идентификаторами.")
    rng = np.random.default_rng(seed)
    categories = sorted(products["category"].unique())
    brands = sorted(products["brand"].unique())
    category_indices = products["category"].map({name: i for i, name in enumerate(categories)}).to_numpy()
    brand_indices = products["brand"].map({name: i for i, name in enumerate(brands)}).to_numpy()
    item_effects = rng.normal(0, 0.35, len(products))
    rows = []
    for index in range(users):
        category_preferences = rng.normal(0, 0.9, len(categories))
        brand_preferences = rng.normal(0, 0.45, len(brands))
        user_bias = rng.normal(0, 0.4)
        count = int(rng.integers(3, min(8, len(products)) + 1))
        # Выбор товаров также зависит от интереса к категории: данные не случайная выборка.
        weights = np.exp(0.45 * category_preferences[category_indices])
        selected = rng.choice(len(products), size=count, replace=False, p=weights / weights.sum())
        for item in selected:
            score = (3.4 + user_bias + category_preferences[category_indices[item]]
                     + brand_preferences[brand_indices[item]] + item_effects[item]
                     + rng.normal(0, 0.7))
            rows.append((f"S{index + 1:04d}", products.iloc[item]["id"], int(np.clip(np.rint(score), 1, 5))))
    return pd.DataFrame(rows, columns=["user_id", "product_id", "rating"]).sort_values(
        ["user_id", "product_id"], ignore_index=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("data/experiment/ratings_synthetic.csv"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Файл уже существует: {args.output}. Укажите другой --output.")
    try:
        ratings = generate(load_products(), args.users, args.seed)
    except ValueError as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="") as file:
        ratings.to_csv(file, index=False)
    print(f"Создано: {args.output}; пользователей: {ratings.user_id.nunique()}, оценок: {len(ratings)}")
