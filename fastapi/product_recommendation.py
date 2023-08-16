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

    def predict(self, data):
        prediction = str(self.classifier.predict(data)[0])
        logging.info(f"[Product_Recommendation] Prediction: {prediction}")
        
        return prediction