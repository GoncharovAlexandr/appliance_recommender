import streamlit as st
import pandas as pd
from pathlib import Path

from recommender.data_loader import load_products, load_ratings
from recommender.engine import RecommendationEngine


# ==========================================
# НАСТРОЙКИ ПРИЛОЖЕНИЯ
# ==========================================

st.set_page_config(
    page_title="HomeTech — подбор бытовой техники",
    page_icon="🏠",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
RATINGS_PATH = BASE_DIR / "data" / "ratings.csv"


# ==========================================
# ЗАГРУЗКА ДАННЫХ
# ==========================================

try:
    products = load_products()
    ratings = load_ratings()

except Exception as error:
    st.error(f"Ошибка загрузки данных: {error}")
    st.stop()


# ==========================================
# ЗАГОЛОВОК
# ==========================================

st.title("🏠 HomeTech")
st.subheader("Рекомендательная система для подбора бытовой техники")

st.write(
    "Подбирайте бытовую технику с учетом ваших предпочтений. "
    "Система анализирует оценки пользователей и предлагает "
    "товары, которые могут вам понравиться."
)

st.divider()


# ==========================================
# БОКОВАЯ ПАНЕЛЬ
# ==========================================

st.sidebar.title("⚙️ Панель управления")

users = sorted(ratings["user_id"].unique().tolist())

user_options = users + ["Новый пользователь"]

selected_user = st.sidebar.selectbox(
    "Выберите пользователя",
    user_options
)

if selected_user == "Новый пользователь":

    st.sidebar.subheader("Создание пользователя")

    new_user_id = st.sidebar.text_input(
        "Введите идентификатор",
        placeholder="Например, U009"
    )

    if new_user_id:
        user_id = new_user_id.strip()
    else:
        user_id = None

else:
    user_id = selected_user

st.sidebar.divider()

st.sidebar.caption(
    "HomeTech — учебный проект рекомендательной системы"
)


# ==========================================
# СОЗДАНИЕ ДВИЖКА РЕКОМЕНДАЦИЙ
# ==========================================

engine = RecommendationEngine(
    products,
    ratings
)


# ==========================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================================

def save_rating(user, product, rating):
    """Добавление или обновление оценки пользователя."""

    current_ratings = load_ratings()

    existing = (
        (current_ratings["user_id"] == user)
        &
        (current_ratings["product_id"] == product)
    )

    if existing.any():

        current_ratings.loc[
            existing, "rating"
        ] = rating

    else:

        new_rating = pd.DataFrame([{
            "user_id": user,
            "product_id": product,
            "rating": rating
        }])

        current_ratings = pd.concat(
            [current_ratings, new_rating],
            ignore_index=True
        )

    current_ratings.to_csv(
        RATINGS_PATH,
        index=False
    )


def category_icon(category):

    icons = {
        "Холодильники": "🧊",
        "Стиральные машины": "🧺",
        "Пылесосы": "🧹",
        "Телевизоры": "📺"
    }

    return icons.get(category, "📦")


def display_products(data):

    if data.empty:
        st.info("Подходящие товары не найдены.")
        return

    columns = st.columns(3)

    for index, (_, product) in enumerate(
        data.iterrows()
    ):

        with columns[index % 3]:

            with st.container(border=True):

                st.subheader(
                    category_icon(product["category"])
                    + " "
                    + product["name"]
                )

                st.caption(
                    f"{product['brand']} · "
                    f"{product['category']}"
                )

                st.write(product["description"])

                st.write(
                    "**Характеристики:** "
                    + str(product["features"])
                )

                st.markdown(
                    f"### {product['price']:,.0f} ₽".replace(
                        ",", " "
                    )
                )

                if "predicted_rating" in product.index:

                    st.metric(
                        "Прогнозируемая оценка",
                        f"{product['predicted_rating']:.2f} / 5"
                    )

                    if "reason" in product.index:
                        st.caption(product["reason"])

                st.caption(
                    f"Артикул: {product['id']}"
                )


# ==========================================
# ВКЛАДКИ ПРИЛОЖЕНИЯ
# ==========================================

tab_catalog, tab_recommendations, tab_ratings = st.tabs([
    "🛍️ Каталог",
    "⭐ Рекомендации",
    "📝 Мои оценки"
])


# ==========================================
# ВКЛАДКА 1. КАТАЛОГ
# ==========================================

with tab_catalog:

    st.header("Каталог бытовой техники")

    col1, col2 = st.columns(2)

    categories = [
        "Все категории"
    ] + sorted(products["category"].unique().tolist())

    with col1:

        selected_category = st.selectbox(
            "Категория техники",
            categories,
            key="catalog_category"
        )

    with col2:

        max_catalog_price = st.slider(
            "Максимальная стоимость",
            min_value=10000,
            max_value=150000,
            value=150000,
            step=5000,
            key="catalog_price"
        )

    filtered_products = products.copy()

    if selected_category != "Все категории":

        filtered_products = filtered_products[
            filtered_products["category"]
            == selected_category
        ]

    filtered_products = filtered_products[
        filtered_products["price"]
        <= max_catalog_price
    ]

    st.write(
        f"Найдено товаров: {len(filtered_products)}"
    )

    display_products(filtered_products)


# ==========================================
# ВКЛАДКА 2. РЕКОМЕНДАЦИИ
# ==========================================

with tab_recommendations:

    st.header("Персональные рекомендации")

    if user_id is None:

        st.warning(
            "Выберите существующего пользователя "
            "или создайте нового в боковой панели."
        )

    else:

        st.write(
            f"Рекомендации для пользователя: **{user_id}**"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            recommendation_category = st.selectbox(
                "Категория",
                categories,
                key="recommendation_category"
            )

        with col2:

            max_price = st.number_input(
                "Максимальная цена (₽)",
                min_value=0,
                max_value=500000,
                value=150000,
                step=5000
            )

        with col3:

            top_n = st.selectbox(
                "Количество рекомендаций",
                [3, 5, 10],
                index=1
            )

        if st.button(
            "🔍 Подобрать технику",
            type="primary",
            use_container_width=True
        ):

            recommendations = engine.recommend(
                user_id=user_id,
                category=recommendation_category,
                max_price=max_price if max_price > 0 else None,
                top_n=top_n
            )

            st.session_state["recommendations"] = (
                recommendations
            )

            st.session_state["recommendation_user"] = (
                user_id
            )

        if (
            "recommendations" in st.session_state
            and st.session_state.get("recommendation_user")
            == user_id
        ):

            recommendations = st.session_state[
                "recommendations"
            ]

            if recommendations.empty:

                st.info(
                    "Не удалось найти подходящие товары. "
                    "Попробуйте изменить параметры поиска."
                )

            else:

                st.success(
                    f"Найдено рекомендаций: {len(recommendations)}"
                )

                display_products(recommendations)

        else:

            st.info(
                "Нажмите кнопку «Подобрать технику», "
                "чтобы получить персональные рекомендации."
            )


# ==========================================
# ВКЛАДКА 3. ОЦЕНКИ
# ==========================================

with tab_ratings:

    st.header("Оценка бытовой техники")

    st.write(
        "Оцените товары, которые вам нравятся. "
        "Ваши оценки будут использоваться "
        "для формирования персональных рекомендаций."
    )

    if user_id is None:

        st.warning(
            "Сначала выберите пользователя "
            "или создайте новый профиль."
        )

    else:

        rated_products = ratings[
            ratings["user_id"] == user_id
        ]

        st.metric(
            "Количество выставленных оценок",
            len(rated_products)
        )

        with st.form("rating_form"):

            selected_product = st.selectbox(
                "Выберите товар",
                products["id"].tolist(),
                format_func=lambda product_id: (
                    products.loc[
                        products["id"] == product_id,
                        "name"
                    ].iloc[0]
                )
            )

            rating_value = st.slider(
                "Ваша оценка",
                min_value=1,
                max_value=5,
                value=5
            )

            submitted = st.form_submit_button(
                "Сохранить оценку",
                use_container_width=True
            )

            if submitted:

                save_rating(
                    user_id,
                    selected_product,
                    rating_value
                )

                st.success(
                    "Оценка успешно сохранена!"
                )

                st.session_state.pop(
                    "recommendations", None
                )

                st.rerun()

        st.divider()

        st.subheader("История оценок")

        if rated_products.empty:

            st.info(
                "Вы пока не оценивали товары."
            )

        else:

            history = rated_products.merge(
                products[["id", "name", "category", "brand"]],
                left_on="product_id",
                right_on="id",
                how="left"
            )

            history = history[
                ["name", "category", "brand", "rating"]
            ]

            history.columns = [
                "Товар",
                "Категория",
                "Производитель",
                "Оценка"
            ]

            st.dataframe(
                history,
                use_container_width=True,
                hide_index=True
            )


# ==========================================
# ПОДВАЛ
# ==========================================

st.divider()

st.caption(
    "HomeTech © 2026 | "
    "Рекомендательная система для подбора бытовой техники"
)
