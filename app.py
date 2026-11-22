import math
import re

import streamlit as st

from recommender.data_loader import load_products, load_ratings
from recommender.engine import RecommendationEngine
from recommender.storage import create_user, load_users, save_rating


st.set_page_config(page_title="HomeTech — подбор техники", page_icon="⌂", layout="wide")

st.markdown("""
<style>
    .block-container {max-width: 1240px; padding-top: 2.5rem; padding-bottom: 4rem;}
    h1, h2, h3 {letter-spacing: -0.035em;}
    .eyebrow {color: inherit; opacity: .75; font-size: .75rem; font-weight: 700; letter-spacing: .13em; text-transform: uppercase; margin-bottom: .35rem;}
    .hero-copy {color: inherit; opacity: .75; font-size: 1.05rem; max-width: 620px; margin-bottom: 1.7rem;}
    .section-note {color: inherit; opacity: .75; margin-top: -.5rem; margin-bottom: 1.2rem;}
    [data-testid="stVerticalBlockBorderWrapper"] > div {border-radius: 14px;}
    [class*="st-key-product_row_"] [data-testid="stHorizontalBlock"] {align-items: stretch !important;}
    [class*="st-key-product_row_"] [data-testid="stColumn"] {align-self: stretch !important; display: flex; flex-direction: column;}
    [class*="st-key-product_row_"] [data-testid="stColumn"] > [data-testid="stVerticalBlock"] {flex: 1;}
    [class*="st-key-product_row_"] [data-testid="stColumn"] > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"] {flex: 1;}
    [class*="st-key-product_row_"] [data-testid="stColumn"] > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"] {height: 100%;}
    @media (min-width: 768px) {
        [class*="st-key-product_row_"] [data-testid="stHeading"] h3 {min-height: calc(2.4em + 28px);}
    }
    [class*="st-key-product_row_"] [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"]:has(> [data-testid="stForm"]) {margin-top: auto;}
    [class*="st-key-product_row_"] [data-testid="stForm"] {border: 0; padding: 0; background: transparent;}
    [data-testid="stTabs"] [role="tablist"] {gap: 2rem;}
    [data-testid="stTabs"] [role="tab"] {height: 3.25rem; font-weight: 600;}
    .stButton button, .stFormSubmitButton button {border-radius: 8px;}
    [data-testid="stSidebar"] [data-testid="InputInstructions"] {display: none;}
    [data-testid="stSidebar"] .block-container {padding-top: 2rem;}
    [data-testid="stMetricValue"] {font-size: 1.35rem;}
</style>
""", unsafe_allow_html=True)

try:
    products = load_products()
    ratings = load_ratings()
    users = load_users(ratings)
except Exception as error:
    st.error(f"Не удалось загрузить данные: {error}")
    st.stop()

if products.empty:
    st.warning("В каталоге пока нет товаров.")
    st.stop()

if "created_user" in st.session_state:
    st.session_state["selected_user"] = st.session_state.pop("created_user")
    st.session_state["new_profile_name"] = ""

if st.session_state.pop("clear_rating_widgets", False):
    for key in list(st.session_state):
        if key.startswith("card_rating_"):
            del st.session_state[key]

st.sidebar.markdown('<p class="eyebrow">Ваше пространство</p>', unsafe_allow_html=True)
st.sidebar.title("Профиль")
st.sidebar.caption("Оценки и рекомендации сохраняются для выбранного профиля.")

def user_label(user):
    return f"{user} · демопрофиль" if re.fullmatch(r"U00[1-8]", user) else user

if users:
    with st.sidebar.container(height=200, border=False):
        user_id = st.radio("Выберите профиль", users, format_func=user_label, key="selected_user")
else:
    user_id = None
    st.sidebar.info("Создайте профиль, чтобы оценивать товары.")

st.sidebar.divider()
st.sidebar.subheader("Новый профиль")
new_user = st.sidebar.text_input("Имя профиля", key="new_profile_name")
if st.sidebar.button("Создать профиль", use_container_width=True):
    try:
        create_user(new_user, ratings)
        st.session_state["created_user"] = new_user.strip()
        st.session_state["profile_created"] = True
        st.rerun()
    except (ValueError, OSError) as error:
        st.sidebar.error(str(error))

if st.session_state.pop("profile_created", False):
    st.sidebar.success("Профиль создан и выбран.")

if user_id:
    user_ratings = ratings[ratings["user_id"] == user_id]
    rating_lookup = user_ratings.set_index("product_id")["rating"].to_dict()
    st.sidebar.divider()
    st.sidebar.caption(f"Ваших оценок: {len(user_ratings)}")
    if not rating_lookup:
        st.sidebar.caption("Оцените несколько товаров — это поможет подобрать рекомендации.")
else:
    rating_lookup = {}

engine = RecommendationEngine(products, ratings)
categories = ["Все категории"] + sorted(products["category"].dropna().unique().tolist())
icons = {"Холодильники": "🧊", "Стиральные машины": "🧺", "Пылесосы": "🧹", "Телевизоры": "📺"}

st.markdown('<p class="eyebrow">Подбор бытовой техники</p>', unsafe_allow_html=True)
st.title("Техника, которая вам подходит")
st.markdown('<p class="hero-copy">Изучайте каталог, оценивайте товары и находите то, что вам понравится.</p>', unsafe_allow_html=True)

saved = st.session_state.pop("rating_saved", None)
if saved and saved[0] == user_id:
    st.success(f"Оценка сохранена: {saved[1]} — {saved[2]} из 5.")


def display_products(data, context, show_rating=True):
    if data.empty:
        st.info("Подходящих товаров не найдено. Попробуйте изменить фильтры.")
        return
    for start in range(0, len(data), 3):
        row = st.container(key=f"product_row_{context}_{start}")
        columns = row.columns(3)
        for column, (_, product) in zip(columns, data.iloc[start:start + 3].iterrows()):
            with column, st.container(border=True):
                st.caption(f"{icons.get(product['category'], '◻')}  {product['category']}  ·  {product['brand']}")
                st.subheader(product["name"])
                st.markdown(f"#### {product['price']:,.0f} ₽".replace(",", " "))
                st.write(product["description"])
                st.caption(str(product["features"]))
                if "predicted_rating" in product.index:
                    reason = product.get("reason", "")
                    label = {"На основе среднего рейтинга товара": "Средняя оценка товара",
                             "На основе среднего рейтинга каталога": "Средняя оценка каталога"}.get(reason, "Прогноз для вас")
                    st.caption(f"{label}: {product['predicted_rating']:.2f} / 5")
                    st.caption(reason)
                previous = rating_lookup.get(product["id"])
                if show_rating and user_id:
                    with st.form(f"rating_form_{context}_{product['id']}"):
                        if previous is not None:
                            st.caption(f"Ваша оценка: {previous:g} / 5")
                        existing = float(previous) if previous is not None else None
                        rating_value = st.selectbox(
                            "Ваша оценка", [1, 2, 3, 4, 5],
                            index=int(existing) - 1 if existing is not None and existing.is_integer() else None,
                            placeholder="Оценить товар", key=f"card_rating_{context}_{user_id}_{product['id']}",
                        )
                        submitted = st.form_submit_button("Сохранить оценку", use_container_width=True)
                    if submitted:
                        if rating_value is None:
                            st.warning("Выберите оценку.")
                        else:
                            try:
                                save_rating(user_id, product["id"], rating_value, products)
                                st.session_state["rating_saved"] = (user_id, product["name"], rating_value)
                                st.session_state["clear_rating_widgets"] = True
                                st.rerun()
                            except (ValueError, OSError) as error:
                                st.error(f"Не удалось сохранить оценку: {error}")


catalog_tab, recommendations_tab, ratings_tab = st.tabs(["Каталог", "Рекомендации", "Мои оценки"])

with catalog_tab:
    st.header("Каталог техники")
    st.markdown('<p class="section-note">Выбирайте категорию и оценивайте знакомые товары.</p>', unsafe_allow_html=True)
    category = st.radio("Категория", categories, horizontal=True, key="catalog_category")
    search_column, price_column = st.columns([2, 1])
    with search_column:
        query = st.text_input("Поиск", placeholder="Название или бренд", key="catalog_search")
    with price_column:
        catalog_price = st.number_input("Цена до, ₽", min_value=0, value=None, placeholder="Без ограничения", step=1000, key="catalog_price")
    filtered = products.copy()
    if category != "Все категории":
        filtered = filtered[filtered["category"] == category]
    if query.strip():
        text_match = filtered["name"].fillna("").str.contains(query.strip(), case=False, regex=False)
        brand_match = filtered["brand"].fillna("").str.contains(query.strip(), case=False, regex=False)
        filtered = filtered[text_match | brand_match]
    if catalog_price is not None:
        filtered = filtered[filtered["price"] <= catalog_price]
    st.caption(f"Найдено товаров: {len(filtered)}")
    display_products(filtered, "catalog")

with recommendations_tab:
    st.header("Подобрано для вас")
    if not user_id:
        st.info("Создайте или выберите профиль, чтобы получить рекомендации.")
    else:
        if not rating_lookup:
            st.info("Пока у вас нет оценок, показываем товары с высоким средним рейтингом. Оцените несколько товаров в каталоге для персонального подбора.")
        else:
            st.markdown('<p class="section-note">Подбор обновляется с учётом ваших оценок.</p>', unsafe_allow_html=True)
        category_column, budget_column = st.columns(2)
        with category_column:
            recommendation_category = st.selectbox("Категория", categories, key="recommendation_category")
        with budget_column:
            max_price = st.number_input("Цена до, ₽", min_value=0, value=None, placeholder="Без ограничения", step=1000, key="recommendation_price")
        recommendations = engine.recommend(user_id, category=recommendation_category, max_price=max_price, top_n=len(products))
        if recommendations.empty:
            st.info("Подходящих рекомендаций нет. Измените фильтры или посмотрите другие категории.")
        else:
            page_size = 3
            pages = math.ceil(len(recommendations) / page_size)
            context = (user_id, recommendation_category, max_price)
            if st.session_state.get("page_context") != context:
                st.session_state["recommendation_page"] = 1
                st.session_state["page_context"] = context
            page = min(st.session_state.get("recommendation_page", 1), pages)
            st.caption(f"Найдено рекомендаций: {len(recommendations)}")
            display_products(recommendations.iloc[(page - 1) * page_size:page * page_size], "recommendations")
            if pages > 1:
                previous_col, page_col, next_col = st.columns([1, 2, 1])
                if previous_col.button("← Назад", disabled=page <= 1, use_container_width=True):
                    st.session_state["recommendation_page"] = page - 1
                    st.rerun()
                page_col.markdown(f"<p style='text-align:center'>Страница {page} из {pages}</p>", unsafe_allow_html=True)
                if next_col.button("Далее →", disabled=page >= pages, use_container_width=True):
                    st.session_state["recommendation_page"] = page + 1
                    st.rerun()

with ratings_tab:
    st.header("Мои оценки")
    if not user_id:
        st.info("Создайте или выберите профиль, чтобы оценивать товары.")
    elif not rating_lookup:
        st.info("Вы ещё не оценивали товары. Откройте каталог и выберите оценку на карточке товара.")
    else:
        st.markdown('<p class="section-note">Здесь можно посмотреть и изменить свои оценки.</p>', unsafe_allow_html=True)
        st.caption(f"Оценено товаров: {len(rating_lookup)}")
        display_products(products[products["id"].isin(rating_lookup)], "rated")
