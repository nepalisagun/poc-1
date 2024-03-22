import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score

def evaluate_model_performance(y_true, y_pred):
    """
    Evaluate the performance of a predictive model using accuracy, precision, and recall.

    Args:
    - y_true: True target values.
    - y_pred: Predicted target values.

    Returns:
    - evaluation_metrics: Dictionary containing evaluation metrics.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')

    evaluation_metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall
    }

    return evaluation_metrics

def plot_predicted_numbers(predictions):
    """
    Plot the predicted Powerball numbers.

    Args:
    - predictions: Predicted Powerball numbers.

    Returns:
    - None
    """
    # Plot histogram of predicted numbers
    plt.figure(figsize=(10, 6))
    sns.histplot(predictions, bins=range(1, 70), kde=True, color='skyblue')
    plt.title("Predicted Powerball Numbers Distribution")
    plt.xlabel("Powerball Number")
    plt.ylabel("Frequency")
    plt.xticks(range(1, 70))
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Assume true_powerball_numbers and predicted_powerball_numbers are available
    # true_powerball_numbers = ...  # True Powerball numbers from the dataset
    # predicted_powerball_numbers = ...  # Predicted Powerball numbers from the model

    # Step 8: Evaluate model performance
    evaluation_metrics = evaluate_model_performance(true_powerball_numbers, predicted_powerball_numbers)
    print("Model Evaluation Metrics:")
    for metric, value in evaluation_metrics.items():
        print(f"{metric.capitalize()}: {value}")

    # Step 9: Plot predicted Powerball numbers
    plot_predicted_numbers(predicted_powerball_numbers)
