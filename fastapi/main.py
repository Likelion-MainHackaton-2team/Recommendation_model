from product_recommendation import ProductRecommend_Classifier
from hashtag_servicer import FeatureExtractor
from budget_analysis import BudgetAnaylsis

from fastapi import FastAPI, Request
from pydantic import BaseModel
from tqdm import tqdm

from typing import Dict, List

import polars as pl
import logging
import pickle

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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
recommender = ProductRecommend_Classifier()
budget_analysis = BudgetAnaylsis()

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
    re_buy: int

class HashtagSimilarityData(BaseModel):
    """Data columns
    hashtag(list): Hashtag, Categorical
    """
    hashtag: list

class BudgeData(BaseModel):
    """Data columns
    date(str): Date, Categorical
    category(str): Category, Categorical
    amount(uint64): Amount, Continuous
    """
    month: list
    category: list
    amount: list

@app.router.get("/product-recommendation")
def product_recommendation_prediction(request: Request):
    """Predicts the sales based on the data provided
    """
    logging.info(f"Product recommendation Requested!")

    data = request.query_params
    data = pl.DataFrame({
        "sales": [data["sales"]],
        "prices": [data["prices"]],
        "VAP": [data["VAP"]],
        "pet_type": [data["pet_type"]],
        # "rating": [data["rating"]],
        "re_buy": [data["re_buy"]]
    })

    return {"prediction": recommender.predict(data)}

@app.router.get("/budget-analysis")
def budget_analysis_prediction(request: Request):
    """Predicts the sales based on the data provided
    """
    logging.info(f"Budget analysis Requested!")
    data = request.query_params

    logging.info(f"Received request: {request}")
    data = {
        "month": data["month"],
        "category": data["category"],
        "amount": data["amount"]
    }

    prediction = budget_analysis.predict(data)
    print(prediction)

    return {budget_analysis.predict(data)}

@app.router.get("/hashtag-similarity")
def hashtag_recommendation_prediction(data: HashtagSimilarityData):
    """Predicts the sales based on the data provided
    """
    logging.info(f"Hashtag similarity Requested!")
    data = data.hashtag
    global hashtag_data

    data = __preprocess_data(data)
    hashtag_data = __preprocess_data(hashtag_data)

    bert_sim_list = []
    for text in tqdm(hashtag_data):
        sim_list = []
        for text_target in data:
            sim_list.append(extractor.get_bert_similarty(text, text_target))
        bert_sim_list.append(sum(sim_list) / len(sim_list))
    
    sim_df = pl.DataFrame({
        "hashtag": hashtag_data,
        "similarity": bert_sim_list
    })
    sim_df = sim_df.sort(by="similarity", descending=True).to_numpy()[:8]
    return {"hashtag_recommendation": sim_df}

@app.get("/")
def index():
    logging.info(f"Index Requested!")
    return {
        "Product Recommendation": {
            "url": "/product-recommendation",
            "data": {
                "sales": "Sales, Integer(Cotinuous)",
                "prices": "Prices, Integer(Cotinuous)",
                "VAP": "VAP, Integer(0 or 1)",
                "pet_type": "Pet type, Integer([0] -> bird, [1] -> cat, [2] -> dog, [3] -> fish, [4] -> hamster, [5] -> rabbit)",
                "rating": "Rating, Integer(0 ~ 10)",
                "re_buy": "Re-buy, Integer(0 or 1)"
            }
        },

        "Hashtag Similarity": {
            "url": "/hashtag-similarity",
            "data": {
                "hashtag": "Hashtag, List[String]"
            }
        },
        
        "Budget Analysis": {
            "url": "/budget-analysis",
            "data": {
                "date": "Date, String",
                "category": "Category, String",
                "amount": "Amount, Integer(Continuous)"
            }
        }
    }

# uvicorn main:app --reload --port 8000 --host 127.0.0.1