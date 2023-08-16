# Recommendation Algorithm using Machine Learning Method

In this repository, we are developing a recommendation algorithm using the Machine Learning approach. Currently, we are in the phase of comparing various datasets, constructing models, and establishing specificity. We kindly ask for your reference in this regard.

## Datasets

The datasets being used are sourced from [Pet Store Records 2020](https://www.kaggle.com/datasets/ippudkiippude/pet-store-records-2020) obtained from Kaggle. Unfortunately, there aren't many datasets available, and even this dataset is quite limited in quantity. Nevertheless, we have decided to use it for now. The preprocessing process involves actions such as removing unnecessary columns and converting categorical data from strings to integer format.

## Model

We are using autoML to discover the best model. Once the model is established, we will utilize [SHAP](https://shap.readthedocs.io/en/latest/) to identify the most influential features of the dataset.

본 Recommendation AI 모델은 우리가 흔히 알고 있는 AI 뿐만 아니라 원초적으로 사용되는 KNN, Clustering 과 같은 Machine Learinng의 기초 영역부터 LLM 모델까지 여러 모델이 사용됩니다 (단, 하이브리드 모델이나 퓨전 모델이 아님을 명시합니다.). 그렇기 때문에 여러 API 호출과 Domain이 존재합니다. 각 인공지능이 하는 역할과 요청하고자 하는 내용이 다르기 때문에 Data Schema도 다릅니다. FastAPI Server 로드 후, docs를 참고하시기 바랍니다.
