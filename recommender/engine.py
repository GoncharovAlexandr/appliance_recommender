
import numpy as np
import pandas as pd

from .matrix import build_preference_matrix
from .similarity import calculate_similarity


class RecommendationEngine:

    def __init__(self, products, ratings):

        self.products = products.copy()
        self.ratings = ratings.copy()

        (
            self.matrix,
            self.users,
            self.product_ids,
            self.user_to_index,
            self.product_to_index
        ) = build_preference_matrix(self.ratings)

        self.similarity_matrix = calculate_similarity(
            self.matrix
        )

        self.user_means = (
            self.ratings.groupby("user_id")["rating"]
            .mean()
            .to_dict()
        )

        self.product_means = (
            self.ratings.groupby("product_id")["rating"]
            .mean()
            .to_dict()
        )

        self.global_mean = (
            float(self.ratings["rating"].mean())
            if not self.ratings.empty
            else 3.0
        )

    def get_popular_products(
        self,
        category=None,
        max_price=None,
        top_n=5
    ):
        """Возвращает популярные товары."""

        products = self.products.copy()

        if category and category != "Все категории":
            products = products[
                products["category"] == category
            ]

        if max_price is not None:
            products = products[
                products["price"] <= max_price
            ]

        products["predicted_rating"] = products[
            "id"
        ].map(self.product_means).fillna(
            self.global_mean
        )

        products["rating_count"] = products[
            "id"
        ].map(
            self.ratings["product_id"].value_counts()
        ).fillna(0)

        products = products.sort_values(
            ["predicted_rating", "rating_count"],
            ascending=False
        )

        products["reason"] = "Популярный товар"

        return products.head(top_n)

    def recommend(
        self,
        user_id,
        category=None,
        max_price=None,
        top_n=5,
        neighbors_count=5
    ):
        """Формирование персональных рекомендаций."""

        # Если пользователь новый,
        # используем рекомендации по популярности.
        if user_id not in self.user_to_index:
            return self.get_popular_products(
                category, max_price, top_n
            )

        user_index = self.user_to_index[user_id]

        user_ratings = self.matrix.getrow(
            user_index
        ).toarray().ravel()

        user_mean = self.user_means.get(
            user_id, self.global_mean
        )

        # Находим ближайших соседей.
        similarities = self.similarity_matrix[
            user_index
        ]

        neighbor_indices = np.argsort(
            similarities
        )[::-1][:neighbors_count]

        # Оставляем только пользователей
        # с положительным сходством.
        neighbors = [
            index
            for index in neighbor_indices
            if similarities[index] > 0
        ]

        # Товары, которые пользователь ещё не оценивал.
        candidates = self.products.copy()

        if category and category != "Все категории":
            candidates = candidates[
                candidates["category"] == category
            ]

        if max_price is not None:
            candidates = candidates[
                candidates["price"] <= max_price
            ]

        rated_ids = {
            product_id
            for product_id, index in self.product_to_index.items()
            if user_ratings[index] > 0
        }

        candidates = candidates[
            ~candidates["id"].isin(rated_ids)
        ]

        results = []

        for _, product in candidates.iterrows():

            product_id = product["id"]

            # Если товара нет в матрице оценок,
            # используем средний рейтинг каталога.
            if product_id not in self.product_to_index:
                predicted = self.global_mean
                reason = "Новый товар в каталоге"

            else:
                product_index = self.product_to_index[
                    product_id
                ]

                weighted_sum = 0.0
                similarity_sum = 0.0

                # Собираем оценки похожих пользователей.
                for neighbor_index in neighbors:

                    neighbor_rating = self.matrix[
                        neighbor_index, product_index
                    ]

                    if neighbor_rating <= 0:
                        continue

                    neighbor_id = self.users[
                        neighbor_index
                    ]

                    similarity = similarities[
                        neighbor_index
                    ]

                    neighbor_mean = self.user_means.get(
                        neighbor_id, self.global_mean
                    )

                    weighted_sum += similarity * (
                        neighbor_rating - neighbor_mean
                    )

                    similarity_sum += abs(similarity)

                if similarity_sum > 0:

                    predicted = (
                        user_mean
                        + weighted_sum / similarity_sum
                    )

                    reason = (
                        "На основе оценок похожих пользователей"
                    )

                else:

                    predicted = self.product_means.get(
                        product_id, self.global_mean
                    )

                    reason = "На основе среднего рейтинга товара"

            predicted = float(
                np.clip(predicted, 1, 5)
            )

            results.append({
                "id": product_id,
                "predicted_rating": round(predicted, 2),
                "reason": reason
            })

        if not results:
            return pd.DataFrame()

        recommendations = pd.DataFrame(results)

        recommendations = recommendations.merge(
            self.products,
            on="id",
            how="left"
        )

        recommendations = recommendations.sort_values(
            "predicted_rating",
            ascending=False
        )

        return recommendations.head(top_n)
