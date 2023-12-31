# Recommendation Algorithm using Machine Learning Method

In this repository, we are developing a recommendation algorithm using the Machine Learning approach. Currently, we are in the phase of comparing various datasets, constructing models, and establishing specificity. We kindly ask for your reference in this regard.

## Datasets

The datasets being used are sourced from [Pet Store Records 2020](https://www.kaggle.com/datasets/ippudkiippude/pet-store-records-2020) obtained from Kaggle. Unfortunately, there aren't many datasets available, and even this dataset is quite limited in quantity. Nevertheless, we have decided to use it for now. The preprocessing process involves actions such as removing unnecessary columns and converting categorical data from strings to integer format.

## Model

We are using autoML to discover the best model. Once the model is established, we will utilize [SHAP](https://shap.readthedocs.io/en/latest/) to identify the most influential features of the dataset.

This Recommendation AI model encompasses a wide spectrum of machine learning approaches, ranging from foundational techniques like K-Nearest Neighbors (KNN) and Clustering to more advanced methods such as Large Language Model (LLM), while excluding hybrid or fusion models. This diverse set of models covers various domains and involves multiple API calls.

Due to the distinct roles and desired outputs of each AI, the data schema varies accordingly. Please consult the FastAPI server's documentation after loading it for further guidance.

If you have any more questions or need clarification, feel free to ask.