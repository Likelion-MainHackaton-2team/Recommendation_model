from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel

import torch

class FeatureExtractor:
    def __init__(self):
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')

    def get_bert_embedding(self, text):
        tokenziers = self.bert_tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(device='cpu')
        with torch.no_grad():
            output = self.bert_model(**tokenziers)
        return output[0][:, 0, :].squeeze(0).numpy()

    def get_bert_similarty(self, text1, text2):
        embedding1 = self.get_bert_embedding(text1)
        embedding2 = self.get_bert_embedding(text2)
        return cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0][0]

    def get_bert_similarty_list(self, text, text_list):
        embedding = self.get_bert_embedding(text)
        embedding_list = [self.get_bert_embedding(t) for t in text_list]
        return cosine_similarity(embedding.reshape(1, -1), embedding_list)[0]
