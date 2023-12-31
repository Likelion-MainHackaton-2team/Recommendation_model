from product_recommendation import ProductRecommend_Classifier
from hashtag_servicer import FeatureExtractor
from budget_analysis import BudgetAnaylsis

from fastapi import FastAPI, Request
from pydantic import BaseModel
from tqdm import tqdm

import mysql.connector as database
import polars as pl
import logging
import pickle
import json
import sys

# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

MODEL_PATH = "./export/exported_dummy_classifier.pkl"
HASHTAG_TRAIN_DATA = "./data/tags.txt"
HASHTAG_TRAIN_DATA_MAP = "./data/tags_map.json"
PRODUCT_MAP = "./data/product_map.json"

with open(MODEL_PATH, "rb") as f:   model = pickle.load(f)
with open(HASHTAG_TRAIN_DATA, "r") as f:    hashtag_data = f.readlines()
with open(HASHTAG_TRAIN_DATA_MAP, "r") as f:    hashtag_map = json.load(f)
with open(PRODUCT_MAP, "r") as f:    product_map = json.load(f)

hashtag_map_dict = {}
for i in range(len(hashtag_data)): 
    value = hashtag_data[i].replace("\n", "")
    hashtag_map_dict[str(i)] = value

reverse_hashtag_map_dict = {}
for key, value in hashtag_map_dict.items(): reverse_hashtag_map_dict[value] = key

product_map_dict = {}
for key, value in product_map.items():    product_map_dict[key] = value

# MariaDB connection
DB_USER = "ai_user"
DB_PASSWORD = "p@ssw0rd"
DB_HOST = "db-i1ue7-kr.vpc-pub-cdb.ntruss.com"
DB_PORT = 3306
DB_DATABASE = "amicadb"

connector = database.connect(
    user=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT,
    database=DB_DATABASE)
cursor = connector.cursor()

def __preprocess_data(data):
    # Input data will be string "{1, 10, 23}"
    if type(data) == str:
        data = data.replace("{", "")
        data = data.replace("}", "")
        data = data.replace(" ", "")
        data = data.split(",")

        data = [hashtag_map_dict[i] for i in data]

    for idx, word in enumerate(data):
        word_temp = str()
        word_temp = word.replace("\n", "")
        word_temp = word_temp.replace(" ", "")

        if word_temp == "" or word_temp == " " or word_temp == "\n":    continue
        data[idx] = word_temp
    return data

def __hashtag_reverse_map(data):
    reverse_hashtag_map_dict = {}

    for key, value in hashtag_map_dict.items():
        reverse_hashtag_map_dict[value] = key
        
    data = [reverse_hashtag_map_dict[i] for i in data]
    return data
    
def __mariadb_query(table_name, query=None):
    if query is None:   query = f"SELECT * FROM {table_name}"

    cursor.execute(query)
    data = cursor.fetchall()

    data = pl.DataFrame(data)

    return data

def __pre_product_recommend_data(pet_type, pet_size):
    pet_type_map = {
        'cat': 0,
        'dog': 1,
    }

    pet_size_map = {
        'extra_small': 0,
        'small' : 1,
        'medium': 2,
        'large': 3,
        'extra_large': 4,
    }

    pet_type = pet_type_map[pet_type]
    pet_size = pet_size_map[pet_size]

    return pet_type, pet_size

app = FastAPI()
extractor = FeatureExtractor()
recommender = ProductRecommend_Classifier()
budget_analysis = BudgetAnaylsis()

class ProductRecommendData(BaseModel):
    species: str
    userPetSize: str

class HashtagSimilarityData(BaseModel):
    """Data columns
    hashtag(list): Hashtag, Categorical"""
    hashtag: str

class BudgeData(BaseModel):
    id: str
    year: int 
    month: int

# DONE
@app.router.get("/product-recommendation")
def product_recommendation_prediction(request: ProductRecommendData):
    """Predicts the sales based on the data provided
    """
    logging.info(f"Product recommendation Requested!")

    user_pet_type = request.species
    user_pet_size = request.userPetSize
    user_pet_type, user_pet_size = __pre_product_recommend_data(user_pet_type, user_pet_size)
    print(user_pet_type, user_pet_size)

    prediction = recommender.predict(user_pet_type, user_pet_size)
    as_string = product_map_dict[str(prediction)]

    return {
        "prediction": prediction,
        "as_string": as_string,
    }

# on progress
@app.router.get("/budget-analysis")
def budget_analysis_prediction(request: BudgeData):
    """Requests
    id(str), year(int), month(int)"""
    logging.info(f"Budget analysis Requested!")
    table_name = "product"
    data = __mariadb_query(
        table_name,
        query=f"SELECT * FROM {table_name} WHERE id = {request.id} AND year = {request.year} AND month = {request.month}"
    )

    id, year, month = request.id, request.year, request.month
    data = budget_analysis.predict(data, id, year, month)

    return {
        "message": "Not implemented yet!"
    }

# DONE
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

    response = pl.DataFrame({
        "hashtag": __hashtag_reverse_map(hashtag_data),
        "similarity": bert_sim_list,
        "original": hashtag_data,
    })

    response = response.sort("similarity", descending=True)
    response = response.to_dict()

    hashtag_list = response["hashtag"].to_list()[:8]
    original_list = response["original"].to_list()[:8]

    return {
        "hashtag": hashtag_list,
        "original": original_list,
    }

@app.get("/")
def index():
    logging.info(f"Index Requested!")
    return {"message": "Hello World"}

# uvicorn main:app --reload --port 8000 --host 127.0.0.1