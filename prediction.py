import pickle
import pandas as pd

def load_model(model_name):
    """
    Load a trained model from disk.
    """
    with open(f"{model_name}.pkl", "rb") as f:
        model = pickle.load(f)
    return model

def predict(gene_info):
    """
    Make a prediction for a given gene.
    """
    # Load the models
    models = {
        'XGBoost': load_model('XGBoost.pkl'),
        'SVM': load_model('SVM.pkl'),
        'Random Forest': load_model('Random Forest.pkl'),
    }

    # Make predictions using each model
    predictions = []
    for model_name, model in models.items():
        prediction = model.predict(gene_info)
        predictions.append(prediction[0])

    # Return the majority vote as the final prediction
    return max(set(predictions), key=predictions.count)
