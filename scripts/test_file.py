import joblib
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from termcolor import colored
from timer_thread import with_timer

def write_to_file(output_path, result):
    with open(output_path, 'w+') as out:
        out.write(result)
def load_model(model_path):
    return joblib.load(model_path)

def load_label_encoder(lencoder_path):
    return joblib.load(lencoder_path)

@with_timer
def load_original_data(original_data_path):
    original_data = pd.read_csv(original_data_path)
    original_data.dropna(inplace=True)
    original_data['NER'] = original_data['NER'].apply(eval)
    return original_data

def get_ingredients(input_path):
    lines = []
    with open(input_path) as f:
        lines = f.read().split('\n')
    return lines

@with_timer
def prepare_for_prediction_and_return_input_df(model, ingredients, original_data):
    all_ingredients = model.estimators_[0].coef_.shape[1] 
    mlb = MultiLabelBinarizer()
    mlb.fit([ingredients]) 

    mlb = MultiLabelBinarizer(sparse_output=True)
    mlb.fit(original_data['NER'])   

    input_vector = mlb.transform([ingredients])  # sparse
    input_df = pd.DataFrame.sparse.from_spmatrix(
        input_vector,
        columns=mlb.classes_
    )
    ingredient_columns = model.classes_.shape[0]  

    for col in model.estimators_[0].feature_names_in_:
        if col not in input_df.columns:
            input_df[col] = 0
    
    input_df = input_df[model.estimators_[0].feature_names_in_]

    return input_df

@with_timer
def run_model_prediction(model, input_df, label_encoder):
    predicted_label = model.predict(input_df)[0]
    predicted_title = label_encoder.inverse_transform([predicted_label])[0]
    return [predicted_label, predicted_title]

def run_model(manifest):
    print(colored("Loading model...", "cyan"))
    model = load_model('recipe_model.pkl')

    print(colored("Loading label encoder...", "cyan"))
    label_encoder = load_label_encoder('label_encoder.pkl')

    print(colored("Loading ingredients...", "cyan"))
    ingredients = get_ingredients(manifest["input"])
    print(ingredients)

    print(colored("Loading original data...", "cyan"))
    original_data = load_original_data('../data/RecipeNLG_dataset.csv')

    print(colored("Preparing for prediction...", "yellow"))
    input_df = prepare_for_prediction_and_return_input_df(model,ingredients,original_data)
    print(colored("Predicting, go get yourself some coffee :)", "light_green"))
    [predicted_label, predicted_title] = run_model_prediction(model,input_df, label_encoder)
    print(type(predicted_label))
    print(type(predicted_title))
    print("Model przewidzał przepis: ", predicted_title)

    write_to_file(manifest["output"], f"{predicted_label}\nNa podstawie podanych składników stwierdzam że najbardziej trafnym daniem będzie: {predicted_title}.\nSmacznego :)")
