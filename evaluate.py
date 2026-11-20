import numpy as np
import pandas as pd

from recommender.data_loader import (
    load_products,
    load_ratings
)

from recommender.engine import RecommendationEngine


def evaluate():

    products = load_products()
    ratings = load_ratings()

    # Фиксируем случайность для повторяемости эксперимента.
    rng = np.random.default_rng(42)

    # Разделяем оценки на обучающую и тестовую выборки.
    train_parts = []
    test_parts = []

    for _, group in ratings.groupby("user_id"):

        group = group.sample(
            frac=1,
            random_state=42
        )

        if len(group) < 2:
            train_parts.append(group)
            continue

        test_count = max(
            1,
            int(len(group) * 0.2)
        )

        test_parts.append(
            group.iloc[:test_count]
        )

        train_parts.append(
            group.iloc[test_count:]
        )

    train = pd.concat(
        train_parts,
        ignore_index=True
    )

    test = pd.concat(
        test_parts,
        ignore_index=True
    )

    engine = RecommendationEngine(
        products,
        train
    )

    actual = []
    predicted = []

    # Проверяем скрытые оценки.
    for _, row in test.iterrows():

        user_id = row["user_id"]
        product_id = row["product_id"]

        user_mean = engine.user_means.get(
            user_id,
            engine.global_mean
        )

        user_index = engine.user_to_index.get(
            user_id
        )

        product_index = engine.product_to_index.get(
            product_id
        )

        prediction = user_mean

        if (
            user_index is not None
            and product_index is not None
        ):

            similarities = engine.similarity_matrix[
                user_index
            ]

            weighted_sum = 0.0
            similarity_sum = 0.0

            neighbor_indices = np.argsort(
                similarities
            )[::-1][:5]

            for neighbor_index in neighbor_indices:

                similarity = similarities[
                    neighbor_index
                ]

                neighbor_rating = engine.matrix[
                    neighbor_index,
                    product_index
                ]

                if similarity <= 0 or neighbor_rating <= 0:
                    continue

                neighbor_id = engine.users[
                    neighbor_index
                ]

                neighbor_mean = engine.user_means.get(
                    neighbor_id,
                    engine.global_mean
                )

                weighted_sum += similarity * (
                    neighbor_rating - neighbor_mean
                )

                similarity_sum += similarity

            if similarity_sum > 0:

                prediction = (
                    user_mean
                    + weighted_sum / similarity_sum
                )

        prediction = np.clip(
            prediction,
            1,
            5
        )

        actual.append(row["rating"])
        predicted.append(prediction)

    actual = np.array(actual)
    predicted = np.array(predicted)

    mae = np.mean(
        np.abs(actual - predicted)
    )

    rmse = np.sqrt(
        np.mean((actual - predicted) ** 2)
    )

    print("\n=== РЕЗУЛЬТАТЫ ОЦЕНКИ ===")

    print("Обучающих оценок:", len(train))
    print("Тестовых оценок:", len(test))

    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")


if __name__ == "__main__":
    evaluate()
