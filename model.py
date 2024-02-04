import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from imblearn.over_sampling import RandomOverSampler

def preprocess_data(genes):
    # Drop unnecessary columns
    columns_to_drop = ['status', 'chromosome', 'number-of-reports', 'gene-name', 'ensembl-id', 'gene-score', 'genetic-category']
    genes = genes.drop(columns=columns_to_drop)

    # Encode gene symbols as dummy variables
    genes_encoded = pd.get_dummies(genes, columns=['gene-symbol'])

    # Features (X) excluding the 'syndromic' column
    X = genes_encoded.drop(columns='syndromic')

    # Labels (y)
    y = genes_encoded['syndromic']

    # Convert to binary classification (1 for syndromic, 0 for non-syndromic)
    y_binary = (y == 1).astype(int)

    # Resample the dataset
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y_binary)

    return X_resampled, y_resampled

def train_and_evaluate_classifiers(X, y):
    classifiers = {
        'XGBoost': XGBClassifier(),
        'SVM': SVC(),
        'Random Forest': RandomForestClassifier()
    }

    for clf_name, clf in classifiers.items():
        # Train the classifier
        clf.fit(X, y)

        # Evaluate the classifier
        y_pred = clf.predict(X)
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)

        print(f"\nResults for {clf_name} on original data:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
 # Classification Report
        report = classification_report(y, y_pred)
        print(report)

def save_models(classifiers):
    """
    Save the trained models to disk.
    """
    for clf_name, clf in classifiers.items():
        filename = f"{clf_name}.pkl"
        with open(filename, "wb") as f:
            pickle.dump(clf, f)

# Train and evaluate the classifiers
X_resampled, y_resampled = preprocess_data(genes)
classifiers = train_and_evaluate_classifiers(X_resampled, y_resampled)

# Save the trained models
save_models(classifiers)
