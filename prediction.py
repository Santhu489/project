import pickle
import pandas as pd

def load_model(model_name):
    """
    Load a trained model from disk.
    """
    with open(f"{model_name}.pkl", "rb") as f:
        model = pickle.load(f)
    return model

def preprocess_input(gene_info):
    # Perform the same preprocessing steps as during training
    columns_to_drop = ['status', 'chromosome', 'number-of-reports', 'gene-name', 'ensembl-id', 'gene-score', 'genetic-category']
    gene_info = gene_info.drop(columns=columns_to_drop)

    # Encode gene symbols as dummy variables
    gene_info_encoded = pd.get_dummies(gene_info, columns=['gene-symbol'])

    # Return the preprocessed DataFrame
    return gene_info_encoded

def predict(gene_info):
    """
    Make a prediction for a given gene.
    """
    # Load the models
    models = {
        'SVM': load_model('SVM_model'),
        'Random Forest': load_model('Random Forest_model'),
    }

    # Preprocess the input data
    gene_info_processed = preprocess_input(gene_info)

    # Make predictions using each model
    predictions = []
    for model_name, model in models.items():
        prediction = model.predict(gene_info_processed)
        predictions.append(prediction[0])

    # Return the majority vote as the final prediction
    return max(set(predictions), key=predictions.count)
