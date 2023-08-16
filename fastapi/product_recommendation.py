from sklearn.dummy import DummyClassifier

import polars as pl
import logging
import pickle

logging.basicConfig(level=logging.INFO)

DUMMY_MODEL_PATH = "./export/exported_dummy_classifier.pkl"
with open(DUMMY_MODEL_PATH, "rb") as f:   model = pickle.load(f)

class ProductRecommend_Classifier:
    def __init__(self):
        self.classifier = model

    def predict(self, data, user_info):
        data = data.filter(
            pl.col["pet_type"] == user_info["pet_type"],
            pl.col["pet_size"] == user_info["pet_size"],
        )

        prediction = self.classifier.predict(data)
        return prediction