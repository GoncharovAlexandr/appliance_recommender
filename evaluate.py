"""Повторяемое сравнение рекомендателя с простым базовым прогнозом."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from recommender.data_loader import load_products, load_ratings
from recommender.engine import RecommendationEngine
from recommender.legacy import LegacyRecommendationEngine


DEFAULT_RATINGS = Path(__file__).resolve().parent / "data" / "experiment" / "ratings_demo.csv"


def split_ratings(ratings, seed):
    """Отложить около 20% оценок каждого пользователя, оставив хотя бы одну."""
    rng = np.random.default_rng(seed)
    train_parts, test_parts = [], []
    for _, group in ratings.groupby("user_id", sort=True):
        group = group.iloc[rng.permutation(len(group))]
        if len(group) < 2:
            train_parts.append(group)
            continue
        count = max(1, int(len(group) * 0.2))
        test_parts.append(group.iloc[:count])
        train_parts.append(group.iloc[count:])
    if not test_parts:
        raise ValueError("Недостаточно оценок: нужен хотя бы один пользователь с двумя оценками.")
    return pd.concat(train_parts, ignore_index=True), pd.concat(test_parts, ignore_index=True)


def evaluate_split(products, ratings, seed, engine_factory=RecommendationEngine, ranking=False):
    train, test = split_ratings(ratings, seed)
    engine = engine_factory(products, train)
    actual = test["rating"].to_numpy(dtype=float)
    outputs = [engine.predict_rating(row.user_id, row.product_id) for row in test.itertuples()]
    predicted = np.array([result[0] for result in outputs])
    baseline = np.array([engine.product_means.get(row.product_id, engine.global_mean) for row in test.itertuples()])

    def errors(values):
        difference = actual - values
        return float(np.mean(np.abs(difference))), float(np.sqrt(np.mean(difference ** 2)))

    mae, rmse = errors(predicted)
    baseline_mae, baseline_rmse = errors(baseline)
    personalized = sum(reason == "На основе оценок похожих пользователей" for _, reason in outputs)
    result = {"seed": seed, "train": len(train), "test": len(test), "mae": mae, "rmse": rmse,
            "baseline_mae": baseline_mae, "baseline_rmse": baseline_rmse,
            "personalized": personalized}
    if ranking:
        hits = baseline_hits = eligible = 0
        for user, group in test.groupby("user_id"):
            relevant = set(group.loc[group.rating >= 4, "product_id"])
            if not relevant:
                continue
            rated = set(train.loc[train.user_id == user, "product_id"])
            candidates = products.loc[~products.id.isin(rated), "id"].tolist()
            predictions = [(item, engine.predict_rating(user, item)[0]) for item in candidates]
            baseline_predictions = [(item, engine.product_means.get(item, engine.global_mean)) for item in candidates]
            top = {item for item, _ in sorted(predictions, key=lambda pair: (-pair[1], pair[0]))[:3]}
            baseline_top = {item for item, _ in sorted(baseline_predictions, key=lambda pair: (-pair[1], pair[0]))[:3]}
            hits += bool(top & relevant)
            baseline_hits += bool(baseline_top & relevant)
            eligible += 1
        result.update(ranking_users=eligible, hits=hits, baseline_hits=baseline_hits)
    return result


def evaluate(ratings_path=DEFAULT_RATINGS, runs=10, seed=42, model="current", ranking=False):
    if runs < 1:
        raise ValueError("Количество разбиений должно быть положительным.")
    products, ratings = load_products(), load_ratings(ratings_path)
    unknown = set(ratings.product_id) - set(products.id)
    if unknown:
        raise ValueError(f"Оценки с неизвестными товарами: {', '.join(sorted(unknown))}")
    if ratings.empty:
        raise ValueError("Набор оценок пуст.")
    engines = {"current": RecommendationEngine, "legacy": LegacyRecommendationEngine}
    if model not in engines:
        raise ValueError("Неизвестная модель")
    results = [evaluate_split(products, ratings, seed + index, engines[model], ranking) for index in range(runs)]
    print(f"Данные: {ratings_path}")
    print(f"Алгоритм: {model}")
    print(f"Товаров: {len(products)}, пользователей: {ratings.user_id.nunique()}, оценок: {len(ratings)}")
    print(f"Разбиений: {runs}, начальное seed: {seed}; тестовых оценок на разбиение: {results[0]['test']}")
    print("Средние метрики по разбиениям (диапазон min–max):")
    for label, key in [("MAE алгоритма", "mae"), ("RMSE алгоритма", "rmse"),
                       ("MAE среднего товара", "baseline_mae"), ("RMSE среднего товара", "baseline_rmse")]:
        values = [result[key] for result in results]
        print(f"  {label}: {np.mean(values):.4f} ({min(values):.4f}–{max(values):.4f})")
    personalized = sum(result["personalized"] for result in results)
    count = sum(result["test"] for result in results)
    print(f"Прогнозы по похожим пользователям: {personalized}/{count} ({personalized / count:.1%})")
    if ranking:
        users = sum(result["ranking_users"] for result in results)
        if users:
            hits = sum(result["hits"] for result in results)
            baseline_hits = sum(result["baseline_hits"] for result in results)
            print(f"Попадание в топ-3: алгоритм {hits}/{users} ({hits / users:.1%}); средний рейтинг {baseline_hits}/{users} ({baseline_hits / users:.1%})")
            print("Учитываются пользователи с отложенной оценкой 4 или 5; неоценённые товары не считаются отрицательными.")
        else:
            print("Попадание в топ-3: нет пользователей с отложенными оценками 4 или 5.")
    print("Разбиения пересекаются; малый искусственный набор не позволяет надёжно оценить качество.")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings", type=Path, default=DEFAULT_RATINGS, help="CSV с оценками")
    parser.add_argument("--runs", type=int, default=10, help="число разбиений (по умолчанию 10)")
    parser.add_argument("--seed", type=int, default=42, help="начальное значение для разбиений")
    parser.add_argument("--model", choices=["current", "legacy"], default="current", help="текущая или прежняя модель")
    parser.add_argument("--ranking", action="store_true", help="проверить попадание высоко оценённых товаров в топ-3")
    args = parser.parse_args()
    try:
        evaluate(args.ratings, args.runs, args.seed, args.model, args.ranking)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f"Ошибка: {error}\n")
