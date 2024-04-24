import numpy as np
from joblib import load
# Importing test data from forest.py
from forest import X_test, y_test

def main():
    # Load the saved RandomForest model
    model = load('random_forest_model.joblib')

    # Make predictions with the loaded model
    predictions = model.predict(X_test)

    # Optionally, calculate and print accuracy if y_test is available
    accuracy = np.mean(predictions == y_test)
    print(f'Accuracy on test data: {accuracy:.4f}')

if __name__ == "__main__":
    main()
