from sklearn.dummy import DummyClassifier

import polars as pl
import logging
import pickle

import torch.nn.functional as F
import torch.nn as nn
import torch

logging.basicConfig(level=logging.INFO)

class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.input_linear_1 = nn.Linear(self.input_dim, 32)
        self.input_linear_1_1 = nn.Linear(32, 64)
        self.input_linear_1_2 = nn.Linear(64, 128)
        self.input_linear_1_3 = nn.Linear(128, 256)
        self.input_linear_1_4 = nn.Linear(256, 512)
        self.dropout_1 = nn.Dropout(0.2)
        self.gru_1 = nn.GRU(512, 512, batch_first=True)
        
        self.input_linear_2 = nn.Linear(self.input_dim, 32)
        self.input_linear_2_1 = nn.Linear(32, 64)
        self.input_linear_2_2 = nn.Linear(64, 128)
        self.input_linear_2_3 = nn.Linear(128, 256)
        self.input_linear_2_4 = nn.Linear(256, 512)
        self.dropout_2 = nn.Dropout(0.2)
        self.gru_2 = nn.GRU(512, 512, batch_first=True)

        self.output_hidden_1 = nn.Linear(1024, 256)
        self.output_hidden_2 = nn.Linear(256, 32)
        self.output_linear = nn.Linear(32, self.output_dim)

    def forward(self, x1, x2):
        x1 = self.input_linear_1(x1)
        x1 = self.input_linear_1_1(x1)
        x1 = self.input_linear_1_2(x1)
        x1 = self.input_linear_1_3(x1)
        x1 = self.input_linear_1_4(x1)
        x1 = self.dropout_1(x1)
        x1, _ = self.gru_1(x1)
    
        x2 = self.input_linear_2(x2)
        x2 = self.input_linear_2_1(x2)
        x2 = self.input_linear_2_2(x2)
        x2 = self.input_linear_2_3(x2)
        x2 = self.input_linear_2_4(x2)
        x2 = self.dropout_2(x2)
        x2, _ = self.gru_2(x2)

        x = torch.cat([x1, x2], dim=1)
        x = F.relu(self.output_hidden_1(x))
        x = F.relu(self.output_hidden_2(x))
        x = self.output_linear(x)

        return x

DUMMY_MODEL_PATH = "./export/exported_dummy_classifier.pkl"
DEEP_LEARNING_MODEL_PATH = "./export/deep_learning_model.h5"
with open(DUMMY_MODEL_PATH, "rb") as f:   model = pickle.load(f)

# deep_learnig = Model(1, 6).to(device='cpu')
# with open(DEEP_LEARNING_MODEL_PATH, "rb") as f: 
#     deep_learnig = deep_learnig.load_state_dict(torch.load(DEEP_LEARNING_MODEL_PATH))

class ProductRecommend_Classifier:
    def __init__(self):
        self.classifier = model

    def predict(self, pet_type, pet_size, deep_learning=False):
        # if deep_learning:
        #     data = [[pet_type, pet_size]]
        #     prediction = self.deep_learnig(data)

        data = [[pet_size, pet_type]]
        prediction = self.classifier.predict(data)

        return prediction.tolist()[0]
    