import pandas as pd
import ace_tools as tools
from name_lib import L2_L3_NAME
import os
import argparse

def performance_calculator(result_dir):
    # Load the provided CSV files
    correctly_classified_path = os.path.join(result_dir, 'correctly_classified_samples_cvtest.csv')
    misclassified_path = os.path.join(result_dir, 'misclassified_samples_cvtest.csv')

    # Read the CSV files
    correctly_classified_df = pd.read_csv(correctly_classified_path)
    misclassified_df = pd.read_csv(misclassified_path)

    # Reverse mapping for quick lookup: L3 to L2
    L3_to_L2 = {l3: l2 for l2, l3_list in L2_L3_NAME.items() for l3 in l3_list}

    # Step 2: Compute TP, FP, FN counts
    L2_metrics = {l2: {'TP': 0, 'FP': 0, 'FN': 0} for l2 in L2_L3_NAME.keys()}

    # True Positives: Count correctly classified samples per L2 class
    for _, row in correctly_classified_df.iterrows():
        l2_label = L3_to_L2.get(row['ground_truth_word_label'])
        if l2_label:
            L2_metrics[l2_label]['TP'] += 1

    # False Negatives and False Positives: Count misclassified samples
    for _, row in misclassified_df.iterrows():
        ground_truth_l2 = L3_to_L2.get(row['ground_truth_word_label'])
        predicted_l2 = L3_to_L2.get(row['predicted_word_label'])

        # False Negative: Model failed to predict this L2 correctly
        if ground_truth_l2:
            L2_metrics[ground_truth_l2]['FN'] += 1

        # False Positive: Model incorrectly predicted this L2
        if predicted_l2:
            L2_metrics[predicted_l2]['FP'] += 1

    # Step 3: Compute Precision, Recall, and F1 Score
    for l2, metrics in L2_metrics.items():
        TP, FP, FN = metrics['TP'], metrics['FP'], metrics['FN']

        # Avoid division by zero
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        L2_metrics[l2].update({'Recall': recall, 'Precision': precision, 'F1 Score': f1_score})

    # Convert results to a DataFrame for better visualization
    l2_metrics_df = pd.DataFrame.from_dict(L2_metrics, orient='index')
    return l2_metrics_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="calculate performance metrics between correctly classified and misclassified samples."
    )

    parser.add_argument(
        "--result-dir",  # You can also pass a UMAP file here.
        type=str,
        default="/data/cs_20192023_swintbase_img384",
        help="Path to the folder that contains the CSV files with classifications results. "
    )

    args = parser.parse_args()

    test_result_dir = args.result_dir
    l2_metrics_df = performance_calculator(test_result_dir)
    tools.display_dataframe_to_user(name="L2 Classification Metrics", dataframe=l2_metrics_df)
