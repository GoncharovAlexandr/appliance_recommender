
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
        self.user_counts = self.ratings.groupby("user_id")["rating"].size().to_dict()

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
            ["predicted_rating", "rating_count", "id"],
            ascending=[False, False, True]
        )

        products["reason"] = np.where(
            products["rating_count"] > 0,
            "На основе среднего рейтинга товара",
            "На основе среднего рейтинга каталога"
        )

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

            predicted, reason = self.predict_rating(user_id, product_id, neighbors_count)
            results.append({
                "id": product_id,
                "predicted_rating": predicted,
                "reason": reason
            })

        if not results:
            return pd.DataFrame(columns=[*self.products.columns, "predicted_rating", "reason"])

        recommendations = pd.DataFrame(results).merge(self.products, on="id", how="left")
        return recommendations.sort_values(["predicted_rating", "id"], ascending=[False, True]).head(top_n)

    def predict_rating(self, user_id, product_id, neighbors_count=5):
        """Прогноз и его основание; одинаково используется в выдаче и оценке качества."""
        if neighbors_count < 0:
            raise ValueError("Количество соседей не может быть отрицательным")
        product_mean = self.product_means.get(product_id, self.global_mean)
        if user_id not in self.user_to_index:
            reason = "На основе среднего рейтинга товара" if product_id in self.product_means else "На основе среднего рейтинга каталога"
            return float(product_mean), reason
        user_index = self.user_to_index[user_id]
        similarities = self.similarity_matrix[user_index]
        user_mean = self.user_means[user_id]
        count = self.user_counts[user_id]
        user_bias = (user_mean - self.global_mean) * count / (count + 5.0)
        baseline = product_mean + user_bias

        if product_id not in self.product_to_index:
            return float(np.clip(baseline, 1, 5)), "Новый товар: на основе ваших оценок и среднего по каталогу"
        product_index = self.product_to_index[product_id]
        product_ratings = self.matrix.getcol(product_index).toarray().ravel()
        eligible = np.flatnonzero((similarities > 0) & (product_ratings > 0))
        neighbors = sorted(eligible, key=lambda i: (-similarities[i], self.users[i]))[:neighbors_count]
        weighted_sum = 0.0
        similarity_sum = 0.0
        for neighbor_index in neighbors:
            similarity = similarities[neighbor_index]
            weighted_sum += similarity * (product_ratings[neighbor_index] - self.user_means[self.users[neighbor_index]])
            similarity_sum += similarity
        if similarity_sum:
            return float(np.clip(baseline + weighted_sum / (similarity_sum + 1.0), 1, 5)), "На основе оценок похожих пользователей"
        return float(np.clip(baseline, 1, 5)), "На основе ваших оценок и среднего рейтинга товара"
