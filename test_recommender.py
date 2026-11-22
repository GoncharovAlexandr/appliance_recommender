import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from recommender.data_loader import load_products, load_ratings
from recommender.engine import RecommendationEngine
from recommender.matrix import build_preference_matrix
from recommender.similarity import calculate_similarity
from recommender.legacy import LegacyRecommendationEngine
from recommender.storage import create_user, load_users, save_rating
from evaluate import DEFAULT_RATINGS, evaluate_split, split_ratings
from generate_experiment_data import generate


class RecommendationTests(unittest.TestCase):
    def setUp(self):
        self.products = load_products()
        self.ratings = load_ratings(DEFAULT_RATINGS)
        self.engine = RecommendationEngine(self.products, self.ratings)

    def test_matrix_and_recommendations(self):
        engine = self.engine
        self.assertEqual(engine.matrix.nnz, len(self.ratings))
        self.assertTrue(np.allclose(engine.similarity_matrix, engine.similarity_matrix.T))
        self.assertTrue(np.allclose(np.diag(engine.similarity_matrix), 0))
        result = engine.recommend("U001", top_n=20)
        rated = set(self.ratings.loc[self.ratings.user_id == "U001", "product_id"])
        self.assertFalse(rated.intersection(result.id))
        for row in result.itertuples():
            self.assertAlmostEqual(row.predicted_rating, engine.predict_rating("U001", row.id)[0], places=2, delta=None)

    def test_filters_and_cold_start(self):
        result = self.engine.recommend("new", category="Пылесосы", max_price=15000)
        self.assertTrue((result.category == "Пылесосы").all())
        self.assertTrue((result.price <= 15000).all())
        self.assertTrue(self.engine.recommend("new", max_price=1).empty)

    def test_empty_ratings(self):
        engine = RecommendationEngine(self.products, self.ratings.iloc[:0])
        self.assertEqual(engine.similarity_matrix.shape, (0, 0))
        self.assertFalse(engine.recommend("new").empty)
        self.assertEqual(engine.predict_rating("new", "P001")[0], 3.0)

    def test_experiment_is_reproducible_and_has_no_overlap(self):
        ratings = load_ratings(DEFAULT_RATINGS)
        train, test = split_ratings(ratings, 42)
        train_pairs = set(zip(train.user_id, train.product_id))
        test_pairs = set(zip(test.user_id, test.product_id))
        self.assertFalse(train_pairs & test_pairs)
        self.assertEqual(len(train) + len(test), len(ratings))
        self.assertTrue(set(test.user_id).issubset(set(train.user_id)))
        self.assertEqual(evaluate_split(self.products, ratings, 42), evaluate_split(self.products, ratings, 42))
        self.assertGreaterEqual(evaluate_split(self.products, ratings, 42)["personalized"], 0)

    def test_experiment_needs_testable_ratings(self):
        ratings = pd.DataFrame([{"user_id": "u", "product_id": "p", "rating": 4}])
        with self.assertRaisesRegex(ValueError, "Недостаточно оценок"):
            split_ratings(ratings, 42)

    def test_ranking_evaluation_and_legacy_model(self):
        ratings = load_ratings(DEFAULT_RATINGS)
        result = evaluate_split(self.products, ratings, 42, LegacyRecommendationEngine, ranking=True)
        self.assertGreater(result["ranking_users"], 0)
        self.assertLessEqual(result["hits"], result["ranking_users"])
        self.assertLessEqual(result["baseline_hits"], result["ranking_users"])
        self.assertGreaterEqual(result["mae"], 0)

    def test_synthetic_data_is_valid_and_reproducible(self):
        first = generate(self.products, users=20, seed=2026)
        second = generate(self.products, users=20, seed=2026)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(first.user_id.nunique(), 20)
        self.assertTrue(first.rating.between(1, 5).all())
        self.assertFalse(first.duplicated(["user_id", "product_id"]).any())
        self.assertTrue(set(first.product_id).issubset(set(self.products.id)))
        self.assertTrue(first.groupby("user_id").size().between(3, 8).all())

    def test_similarity_uses_agreement_and_requires_shared_evidence(self):
        ratings = pd.DataFrame([
            ("a", "p1", 5), ("a", "p2", 1),
            ("b", "p1", 4), ("b", "p2", 2),
            ("c", "p1", 1), ("c", "p2", 5),
            ("d", "p1", 5),
        ], columns=["user_id", "product_id", "rating"])
        matrix, _, _, users, _ = build_preference_matrix(ratings)
        similarity = calculate_similarity(matrix)
        self.assertAlmostEqual(similarity[users["a"], users["b"]], 0.25)
        self.assertAlmostEqual(similarity[users["a"], users["c"]], -0.25)
        self.assertEqual(similarity[users["a"], users["d"]], 0)
        self.assertTrue(np.isfinite(similarity).all())

    def test_neighbors_are_selected_among_users_who_rated_target(self):
        ratings = pd.DataFrame([
            ("a", "p1", 5), ("a", "p2", 1),
            ("b", "p1", 5), ("b", "p2", 1),
            ("c", "p1", 4), ("c", "p2", 2), ("c", "p3", 5),
        ], columns=["user_id", "product_id", "rating"])
        products = pd.DataFrame({"id": ["p1", "p2", "p3"], "category": ["x"] * 3, "price": [1] * 3})
        engine = RecommendationEngine(products, ratings)
        predicted, reason = engine.predict_rating("a", "p3", neighbors_count=1)
        self.assertEqual(reason, "На основе оценок похожих пользователей")
        self.assertGreaterEqual(predicted, 1)
        self.assertLessEqual(predicted, 5)
        self.assertEqual(engine.predict_rating("a", "p3", neighbors_count=0)[1],
                         "На основе ваших оценок и среднего рейтинга товара")

    def test_cold_start_explanations_distinguish_unrated_products(self):
        products = pd.DataFrame({"id": ["p1", "p2"], "category": ["x", "x"], "price": [1, 2]})
        ratings = pd.DataFrame([("a", "p1", 4)], columns=["user_id", "product_id", "rating"])
        engine = RecommendationEngine(products, ratings)
        result = engine.recommend("new", top_n=10).set_index("id")
        self.assertEqual(result.loc["p1", "reason"], "На основе среднего рейтинга товара")
        self.assertEqual(result.loc["p2", "reason"], "На основе среднего рейтинга каталога")

    def test_profile_and_rating_persistence(self):
        with tempfile.TemporaryDirectory() as directory:
            users_path = Path(directory) / "users.json"
            ratings_path = Path(directory) / "ratings.csv"
            self.ratings.to_csv(ratings_path, index=False)
            create_user("U011", self.ratings, users_path)
            self.assertIn("U011", load_users(self.ratings, users_path))
            with self.assertRaises(ValueError):
                create_user("U011", self.ratings, users_path)
            save_rating("U011", "P001", 5, self.products, ratings_path)
            save_rating("U011", "P001", 2, self.products, ratings_path)
            rows = load_ratings(ratings_path).query('user_id == "U011" and product_id == "P001"')
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows.iloc[0].rating, 2)
            with self.assertRaises(ValueError):
                save_rating("U011", "P001", 6, self.products, ratings_path)

    def test_created_profile_appears_and_is_selected(self):
        from streamlit.testing.v1 import AppTest

        with tempfile.TemporaryDirectory() as directory:
            users_path = Path(directory) / "users.json"
            def temporary_load(ratings, path=None):
                return load_users(ratings, users_path)
            def temporary_create(user, ratings):
                return create_user(user, ratings, users_path)
            with patch("recommender.data_loader.load_ratings", return_value=self.ratings), \
                 patch("recommender.storage.load_users", temporary_load), patch("recommender.storage.create_user", temporary_create):
                app = AppTest.from_file("app.py").run()
                app.sidebar.text_input[0].input("U011")
                app.sidebar.button[0].click().run()
                self.assertFalse(app.exception)
                self.assertIn("U011", app.radio(key="selected_user").options)
                self.assertEqual(app.radio(key="selected_user").value, "U011")
                app.run()
                self.assertEqual(app.radio(key="selected_user").value, "U011")

    def test_price_zero_and_profile_input(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file("app.py").run()
        app.number_input(key="recommendation_price").set_value(0).run()
        self.assertTrue(any("Подходящих рекомендаций нет" in item.value for item in app.info))
        selected = app.radio(key="selected_user").value
        app.sidebar.text_input[0].input("U001").run()
        self.assertEqual(app.radio(key="selected_user").value, selected)
        self.assertFalse(app.exception)

    def test_catalog_filters_and_recommendation_pagination(self):
        from streamlit.testing.v1 import AppTest
        with patch("recommender.data_loader.load_ratings", return_value=self.ratings):
            app = AppTest.from_file("app.py").run()
            app.radio(key="catalog_category").set_value("Пылесосы").run()
            app.text_input(key="catalog_search").set_value("LG").run()
            self.assertTrue(any("Найдено товаров: 1" in item.value for item in app.caption))
            app.number_input(key="catalog_price").set_value(10000).run()
            self.assertTrue(any("Найдено товаров: 0" in item.value for item in app.caption))
            next(button for button in app.button if button.label == "Далее →").click().run()
            self.assertEqual(app.session_state["recommendation_page"], 2)
            app.selectbox(key="recommendation_category").select("Пылесосы").run()
            self.assertEqual(app.session_state["recommendation_page"], 1)
            self.assertFalse(app.exception)

    def test_existing_rating_is_loaded_for_selected_product(self):
        from streamlit.testing.v1 import AppTest
        ratings = pd.DataFrame([
            {"user_id": "U001", "product_id": "P001", "rating": 2},
            {"user_id": "U001", "product_id": "P002", "rating": 4},
        ])
        with patch("recommender.data_loader.load_ratings", return_value=ratings):
            app = AppTest.from_file("app.py").run()
            self.assertEqual(app.selectbox(key="card_rating_catalog_U001_P001").value, 2)
            self.assertEqual(app.selectbox(key="card_rating_catalog_U001_P002").value, 4)
            self.assertIsNone(app.selectbox(key="card_rating_catalog_U001_P003").value)
            self.assertFalse(app.exception)

    def test_rating_saves_and_updates_through_app(self):
        from streamlit.testing.v1 import AppTest
        with tempfile.TemporaryDirectory() as directory:
            ratings_path = Path(directory) / "ratings.csv"
            users_path = Path(directory) / "users.json"
            self.ratings.to_csv(ratings_path, index=False)
            create_user("CHECK_PROFILE", self.ratings, users_path)

            def temporary_users(ratings, path=None):
                return load_users(ratings, users_path)

            def temporary_save(user, product, rating, products):
                return save_rating(user, product, rating, products, ratings_path)

            with patch("recommender.data_loader.load_ratings", side_effect=lambda: load_ratings(ratings_path)), \
                 patch("recommender.storage.load_users", temporary_users), \
                 patch("recommender.storage.save_rating", temporary_save):
                app = AppTest.from_file("app.py").run()
                app.radio(key="selected_user").set_value("CHECK_PROFILE").run()
                rating = app.selectbox(key="card_rating_catalog_CHECK_PROFILE_P001")
                self.assertIsNone(rating.value)
                app.button(key="FormSubmitter:rating_form_catalog_P001-Сохранить оценку").click().run()
                self.assertTrue(any("Выберите оценку" in warning.value for warning in app.warning))
                rating = app.selectbox(key="card_rating_catalog_CHECK_PROFILE_P001")
                rating.select(2)
                app.button(key="FormSubmitter:rating_form_catalog_P001-Сохранить оценку").click().run()
                rows = load_ratings(ratings_path).query('user_id == "CHECK_PROFILE" and product_id == "P001"')
                self.assertEqual(rows.iloc[0].rating, 2)
                self.assertTrue(any("Оценка сохранена" in item.value for item in app.success))
                rating = app.selectbox(key="card_rating_catalog_CHECK_PROFILE_P001")
                self.assertEqual(rating.value, 2)
                rating.select(4)
                app.button(key="FormSubmitter:rating_form_catalog_P001-Сохранить оценку").click().run()
                rows = load_ratings(ratings_path).query('user_id == "CHECK_PROFILE" and product_id == "P001"')
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows.iloc[0].rating, 4)
                self.assertFalse(app.exception)


if __name__ == "__main__":
    unittest.main()
