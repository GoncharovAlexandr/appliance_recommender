
import numpy as np
from scipy.sparse import csr_matrix


def build_preference_matrix(ratings):
    """
    Формирование разреженной матрицы предпочтений.

    Строки — пользователи.
    Столбцы — товары.
    Значения — оценки пользователей.
    """

    users = sorted(ratings["user_id"].unique())
    products = sorted(ratings["product_id"].unique())

    user_to_index = {
        user: index
        for index, user in enumerate(users)
    }

    product_to_index = {
        product: index
        for index, product in enumerate(products)
    }

    row_indices = ratings["user_id"].map(
        user_to_index
    ).to_numpy()

    col_indices = ratings["product_id"].map(
        product_to_index
    ).to_numpy()

    values = ratings["rating"].to_numpy(dtype=float)

    matrix = csr_matrix(
        (values, (row_indices, col_indices)),
        shape=(len(users), len(products)),
        dtype=np.float64
    )

    return (
        matrix,
        users,
        products,
        user_to_index,
        product_to_index
    )
