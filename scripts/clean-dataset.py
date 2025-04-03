import pandas as pd
import ast
import re

input_path='../data/RecipeNLG_dataset.csv'
output_path='../data/train-data.csv'

print('Reading dataset...')
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
data['ingredients'].map(type).unique()
unique_ingredients = set(ingredient for row in data['ingredients'] for ingredient in row)
ingredient_columns = {}
for ingredient in unique_ingredients:
    ingredient_columns[ingredient] = data['ingredients'].apply(lambda x: 1 if ingredient in x else 0)
data = pd.concat([data, pd.DataFrame(ingredient_columns)], axis=1)
print(f'Final shape: {data.shape}')

print('Exporting new dataset...')
data.to_csv(output_path)
print(f'New dataset exported to {output_path}')