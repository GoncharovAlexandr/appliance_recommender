
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def calculate_similarity(matrix):
    """
    Вычисление матрицы сходства пользователей.

    На вход подается разреженная матрица.
    На выходе — обычная матрица сходства.
    """

    similarity_matrix = cosine_similarity(
        matrix,
        dense_output=True
    )

    # Сходство пользователя с самим собой
    # не должно учитываться при рекомендациях.
    np.fill_diagonal(similarity_matrix, 0)

    return similarity_matrix
