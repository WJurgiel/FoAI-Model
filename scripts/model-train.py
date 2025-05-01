import pandas as pd
import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi
import ast
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from scipy.sparse import csr_matrix
import numpy as np
import joblib 
from sklearn.preprocessing import LabelEncoder

api=KaggleApi()
api.authenticate()

api.dataset_download_file(
    'paultimothymooney/recipenlg',
    file_name='RecipeNLG_dataset.csv',
    path='../data'
)

original_path = '../data/RecipeNLG_dataset.csv'
zip_path = '../data/RecipeNLG_dataset.zip'
os.rename(original_path, zip_path)
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall('../data')
os.remove(zip_path)

input_path='../data/RecipeNLG_dataset.csv'
output_path='../data/train-data.csv'

print(f'Reading dataset from {input_path}')
data = pd.read_csv(input_path)
print(f'Dataset imported. Shape: {data.shape}')

print(f'Cleaning dataset...')
data.dropna(inplace=True)
data.drop_duplicates(inplace=True)
columns_to_drop = ["Unnamed: 0", "ingredients", "directions", "link", "source"]
data.drop(columns=columns_to_drop, inplace=True)
data.rename(columns={"NER": "ingredients"}, inplace=True)
print(f'Shape: {data.shape}')

print('Modifing dataset...')
data['ingredients'] = data['ingredients'].apply(ast.literal_eval)
mlb = MultiLabelBinarizer(sparse_output=True)
ingredient_matrix = mlb.fit_transform(data['ingredients'])
ingredient_df = pd.DataFrame.sparse.from_spmatrix(
    ingredient_matrix, 
    index=data.index, 
    columns=mlb.classes_
)
data = pd.concat([data, ingredient_df], axis=1)
data.drop(columns=['ingredients'], inplace=True)
print(f'Final shape: {data.shape}')

# print('Exporting new dataset...')
# data.to_csv(output_path)
# print(f'New dataset exported to {output_path}')

print('Preparation to training...')
top_recipes = data['title'].value_counts().nlargest(1000).index
data = data[data['title'].isin(top_recipes)]
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['title'])
ingredient_columns = data.columns.difference(['title', 'label'])
X = data[ingredient_columns]
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print('Model training...')
model = LogisticRegression(max_iter=1000)
classifier = OneVsRestClassifier(model)
classifier.fit(X_train, y_train)

print('Saving a model...')
joblib.dump(classifier, 'recipe_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')