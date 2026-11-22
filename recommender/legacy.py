"""Прежний алгоритм, сохранённый только для воспроизводимого сравнения."""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .engine import RecommendationEngine


class LegacyRecommendationEngine(RecommendationEngine):
    def __init__(self, products, ratings):
        super().__init__(products, ratings)
        if self.matrix.shape[0]:
            self.similarity_matrix = cosine_similarity(self.matrix)
            np.fill_diagonal(self.similarity_matrix, 0)

    def predict_rating(self, user_id, product_id, neighbors_count=5):
        if user_id not in self.user_to_index:
            return float(self.product_means.get(product_id, self.global_mean)), "На основе среднего рейтинга товара"
        if product_id not in self.product_to_index:
            return self.global_mean, "Новый товар в каталоге"
        user_index, product_index = self.user_to_index[user_id], self.product_to_index[product_id]
        similarities = self.similarity_matrix[user_index]
        weighted_sum, similarity_sum = 0.0, 0.0
        for index in np.argsort(similarities)[::-1][:neighbors_count]:
            rating = self.matrix[index, product_index]
            if similarities[index] > 0 and rating > 0:
                weighted_sum += similarities[index] * (rating - self.user_means[self.users[index]])
                similarity_sum += similarities[index]
        if similarity_sum:
            value = self.user_means[user_id] + weighted_sum / similarity_sum
            return float(np.clip(value, 1, 5)), "На основе оценок похожих пользователей"
        return float(self.product_means.get(product_id, self.global_mean)), "На основе среднего рейтинга товара"
