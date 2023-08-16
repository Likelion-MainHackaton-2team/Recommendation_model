from product_recommendation import ProductRecommend_Classifier
from hashtag_servicer import FeatureExtractor
from budget_analysis import BudgetAnaylsis

from fastapi import FastAPI, Request
from pydantic import BaseModel
from tqdm import tqdm

from typing import Dict, List

import polars as pl
import logging
import mariadb
import pickle
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

MODEL_PATH = "./export/exported_dummy_classifier.pkl"
HASHTAG_TRAIN_DATA = "./data/tags.txt"
with open(MODEL_PATH, "rb") as f:   model = pickle.load(f)
with open(HASHTAG_TRAIN_DATA, "r") as f:    hashtag_data = f.readlines()

# MariaDB connection
DB_USER = "ai_user"
DB_PASSWORD = "p@ssw0rd"
DB_HOST = "db-i1ue7-kr.vpc-pub-cdb.ntruss.com"
DB_PORT = 3306
DB_DATABASE = "amicadb"

try:
    connector = mariadb.connect(
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
        database=DB_DATABASE)
    cursor = connector.cursor()
except mariadb.Error as e:
    logging.error(f"Error connecting to MariaDB Platform: {e}")
    sys.exit(1)

def __preprocess_data(data):
    for idx, word in enumerate(data):
        word_temp = str()
        word_temp = word.replace("\n", "")
        word_temp = word_temp.replace(" ", "")

        if word_temp == "" or word_temp == " " or word_temp == "\n":    continue
        data[idx] = word_temp
    return data

def __mariadb_query(table_name):
    query = f"SELECT * FROM {table_name}"
    cursor.execute(query)
    data = cursor.fetchall()

    data = pl.DataFrame(data)

    return data

app = FastAPI()
extractor = FeatureExtractor()
recommender = ProductRecommend_Classifier()
budget_analysis = BudgetAnaylsis()

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

    user_info = request.query_params
    data = __mariadb_query("product")

    prediction = recommender.predict(data, user_info)
    return {"prediction": prediction}

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
                "WIP": "Work in progress"
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
        },
    }

# uvicorn main:app --reload --port 8000 --host 127.0.0.1