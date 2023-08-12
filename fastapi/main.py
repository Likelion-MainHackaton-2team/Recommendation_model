from sklearn.dummy import DummyClassifier
from pydantic import BaseModel
from fastapi import FastAPI

import polars as pl
import numpy as np
import pickle
import json

MODEL_PATH = "./export/exported_dummy_classifier.pkl"
with open(MODEL_PATH, "rb") as f:   model = pickle.load(f)

app = FastAPI()

class RecommendationData(BaseModel):
    """Data columns
    sales(uint64): Sales, Continuous
    prices(uint64): Prices, Continuous
    VAP(uint64): VAP, Binary Categorical
    pet_type(uint64): Pet type, Categorical
    rating(uint64): Rating, Categorical 
    re_buy(uint64): Re-buy, Binary Categorical
    """
    sales: int
    prices: int
    VAP: int
    pet_type: int
    rating: int
    re_buy: int

class HashtagSimilarityData(BaseModel):
    """Data columns
    hashtag(list): Hashtag, Categorical
    """
    hashtag: list

@app.post("/product-recommendation")
def product_recommendation_prediction(data: RecommendationData):
    """Predicts the sales based on the data provided
    """
    data = pl.DataFrame({
        "sales": [data.sales],
        "prices": [data.prices],
        "VAP": [data.VAP],
        "pet_type": [data.pet_type],
        "rating": [data.rating],
        "re_buy": [data.re_buy]
    })
    return {"prediction": model.predict(data).to_numpy()[0]}

# @app.post("/hashtag-similarity")
# def hashtag_recommendation_prediction(data: HashtagSimilarityData):

@app.get("/")
def index():
    return {"greeting": "Hello World"}

# uvicorn main:app --reload