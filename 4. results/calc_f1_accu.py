import sys
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

def calculate_metrics(filename):
    # Read the CSV file
    data = pd.read_csv(filename)

    # Extract true labels and predicted labels
    true_labels = data['True Labels']
    predicted_labels = data['Predicted Labels']

    # Calculate F1 score and Accuracy
    f1 = f1_score(true_labels, predicted_labels, average = 'weighted')
    accuracy = accuracy_score(true_labels, predicted_labels)

    return f1, accuracy

if __name__ == "__main__":
    # Check if filename argument is given
    if len(sys.argv) < 2:
        print("Usage: python script.py <filename>")
        sys.exit(1)

    # Calculate metrics
    f1, accuracy = calculate_metrics(sys.argv[1])

    # Print results
    print(f"F1 Score: {f1}")
    print(f"Accuracy: {accuracy}")

