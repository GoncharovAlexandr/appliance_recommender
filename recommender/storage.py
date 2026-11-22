"""Локальное хранение профилей и оценок учебного приложения."""
import json
import os
import tempfile
from pathlib import Path

import pandas as pd

from .data_loader import DATA_DIR, load_ratings


def _atomic_write(path, content):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as file:
        temporary = file.name
        try:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def load_users(ratings=None, path=None):
    path = Path(path or DATA_DIR / "users.json")
    users = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    if not isinstance(users, list) or any(not isinstance(user, str) or not user.strip() for user in users):
        raise ValueError("Некорректный файл профилей")
    ratings = load_ratings() if ratings is None else ratings
    return sorted(set(users) | set(ratings["user_id"]))


def create_user(user, ratings=None, path=None):
    user = user.strip()
    if not user or user == "Новый пользователь":
        raise ValueError("Введите корректный идентификатор пользователя")
    users = load_users(ratings, path)
    if user in users:
        raise ValueError("Пользователь с таким идентификатором уже существует")
    path = Path(path or DATA_DIR / "users.json")
    _atomic_write(path, json.dumps(sorted(users + [user]), ensure_ascii=False, indent=2) + "\n")


def save_rating(user, product, rating, products, path=None):
    if not isinstance(user, str) or not user.strip() or product not in set(products["id"]):
        raise ValueError("Неизвестный пользователь или товар")
    if isinstance(rating, bool) or not isinstance(rating, (int, float)) or not 1 <= rating <= 5:
        raise ValueError("Оценка должна быть от 1 до 5")
    path = Path(path or DATA_DIR / "ratings.csv")
    ratings = load_ratings(path)
    mask = (ratings["user_id"] == user.strip()) & (ratings["product_id"] == product)
    if mask.any():
        ratings.loc[mask, "rating"] = rating
    else:
        ratings = pd.concat([ratings, pd.DataFrame([{"user_id": user.strip(), "product_id": product, "rating": rating}])], ignore_index=True)
    _atomic_write(path, ratings.to_csv(index=False))
