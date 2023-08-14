from hashtag_servicer import FeatureExtractor
from sklearn.dummy import DummyClassifier
from pydantic import BaseModel
from fastapi import FastAPI
from tqdm import tqdm

import polars as pl
import numpy as np
import pickle
import json

MODEL_PATH = "./export/exported_dummy_classifier.pkl"
HASHTAG_TRAIN_DATA = "./data/tags.txt"
with open(MODEL_PATH, "rb") as f:   model = pickle.load(f)
with open(HASHTAG_TRAIN_DATA, "r") as f:    hashtag_data = f.readlines()

def __preprocess_data(data):
    for idx, word in enumerate(data):
        word_temp = str()
        word_temp = word.replace("\n", "")
        word_temp = word_temp.replace(" ", "")

        if word_temp == "" or word_temp == " " or word_temp == "\n":    continue
        data[idx] = word_temp
    return data

app = FastAPI()
extractor = FeatureExtractor()

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
    return {"product_recommendation": model.predict(data).to_numpy()[0]}

@app.post("/hashtag-similarity")
def hashtag_recommendation_prediction(data: HashtagSimilarityData):
    data = data.hashtag
    global hashtag_data

    data = __preprocess_data(data)
    hashtag_data = __preprocess_data(hashtag_data)

    bert_sim_list = []
    for text in tqdm(hashtag_data):
        for text_target in data:
            bert_sim_list.append(extractor.get_bert_similarty(text, text_target))
    
    sim_df = pl.DataFrame({
        "hashtag": hashtag_data,
        "similarity": bert_sim_list
    })
    sim_df = sim_df.sort(by="similarity", descending=True).to_numpy()
    return {"hashtag_recommendation": sim_df}

@app.get("/")
def index():
    return {
        "Product Recommendation": "/product-recommendation", 
        "Hashtag Similarity": "/hashtag-similarity"
    }
# uvicorn main:app --reload