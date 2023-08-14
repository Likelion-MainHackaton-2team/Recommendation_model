from sklearn.dummy import DummyClassifier

import polars as pl
import pickle

DUMMY_MODEL_PATH = "./export/exported_dummy_classifier.pkl"
with open(DUMMY_MODEL_PATH, "rb") as f:   model = pickle.load(f)

class ProductRecommend_Classifier:
    def __init__(self):
        self.classifier = model

    def predict(self, data):
        data = pl.DataFrame({
            "sales": [data.sales],
            "prices": [data.prices],
            "VAP": [data.VAP],
            "pet_type": [data.pet_type],
            # "rating": [data.rating],
            "re_buy": [data.re_buy]
        })

        return self.classifier.predict(data).to_list()[0]