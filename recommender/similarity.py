"""Сходство пользователей по общим оценённым товарам."""
import numpy as np
from scipy.sparse import diags


def calculate_similarity(matrix, shrinkage=3.0):
    """Центрированное косинусное сходство с поправкой на число общих оценок.

    Среднее каждого пользователя вычисляется по всем его оценкам, а скалярное
    произведение и нормы — только по общим товарам. Одной общей оценки
    недостаточно для надёжного сходства. Результат — плотная матрица в памяти.
    """
    if shrinkage < 0:
        raise ValueError("Коэффициент сглаживания не может быть отрицательным")
    matrix = matrix.tocsr().astype(float)
    if matrix.shape[0] == 0:
        return np.empty((0, 0))
    observed = matrix.copy()
    observed.data[:] = 1.0
    counts = np.asarray(observed.sum(axis=1)).ravel()
    means = np.divide(np.asarray(matrix.sum(axis=1)).ravel(), counts,
                      out=np.zeros_like(counts), where=counts > 0)
    centered = matrix - diags(means) @ observed
    dots = (centered @ centered.T).toarray()
    overlap = (observed @ observed.T).toarray()
    norms = (centered.multiply(centered) @ observed.T).toarray()
    denominator = np.sqrt(norms * norms.T)
    similarity = np.divide(dots, denominator, out=np.zeros_like(dots), where=denominator > 1e-12)
    support = np.maximum(overlap - 1, 0)
    weights = np.divide(support, support + shrinkage, out=np.zeros_like(support), where=support + shrinkage > 0)
    similarity *= weights
    np.fill_diagonal(similarity, 0)
    return np.clip(similarity, -1, 1)
