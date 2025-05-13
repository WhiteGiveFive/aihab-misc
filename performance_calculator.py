"""
This program shows the individual Precision, Recall, and F1 score based on L2 labels.
"""

import pandas as pd
# import ace_tools as tools
from name_lib import L2_L3_NAME, REASSIGN_LABEL_NAME_L3
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, cohen_kappa_score


def ck_score(correct_table, misclassified_table):
    # We only care about the numeric labels for κ
    # (you could also use the word labels if you prefer)
    y_true_correct = correct_table['ground_truth_num_label'].astype(int)
    y_pred_correct = correct_table['predicted_label'].astype(int)

    y_true_wrong = misclassified_table['ground_truth_num_label'].astype(int)
    y_pred_wrong = misclassified_table['predicted_label'].astype(int)

    # Concatenate correct + wrong to get the full test set
    y_true = pd.concat([y_true_correct, y_true_wrong], ignore_index=True)
    y_pred = pd.concat([y_pred_correct, y_pred_wrong], ignore_index=True)

    # Compute Cohen's kappa
    kappa = cohen_kappa_score(y_true, y_pred)
    return kappa

def performance_calculator(result_dir):
    # Load the provided CSV files
    correctly_classified_path = os.path.join(result_dir, 'correctly_classified_samples_cvtest.csv')
    misclassified_path = os.path.join(result_dir, 'misclassified_samples_cvtest.csv')

    # Read the CSV files
    correctly_classified_df = pd.read_csv(correctly_classified_path)
    misclassified_df = pd.read_csv(misclassified_path)

    # Calculate the Cohen's Kappa score
    kappa_score = ck_score(correctly_classified_df, misclassified_df)

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

    # Step 3: Compute Precision, Recall, F1 Score, and Total counts for each L2 class
    for l2, metrics in L2_metrics.items():
        TP, FP, FN = metrics['TP'], metrics['FP'], metrics['FN']

        # Avoid division by zero
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Total ground truth samples for this L2 class
        total = TP + FN

        L2_metrics[l2].update({
            'Recall': recall,
            'Precision': precision,
            'F1 Score': f1_score,
            'Total': total
        })

    # Convert results to a DataFrame for better visualization
    l2_metrics_df = pd.DataFrame.from_dict(L2_metrics, orient='index')
    return l2_metrics_df, kappa_score


def plot_cm(cm_path, class_names: list, normalized: bool = False) -> None:
    def _custom_format(x):
        """
        Custom formatting for the confusion matrix.
        If an entry is smaller than 0.01 in absolute value, it returns '0';
        otherwise, it returns the value formatted to 2 decimal places.
        """
        if abs(x) < 0.01:
            return '0'
        else:
            return f'{x:.2f}'

    cm = np.load(cm_path)
    # Set up annotations based on whether it's normalized or not
    if normalized:
        annot_data = np.array([[_custom_format(val) for val in row] for row in cm])
        fmt = ''
    else:
        annot_data = cm.astype(int)  # Convert to integer format for non-normalized matrix
        fmt = 'd'

    # ---------------------
    # Plot the full confusion matrix
    # ---------------------
    plt.figure(figsize=(15, 12))
    ax = sns.heatmap(cm, annot=annot_data, fmt=fmt, cmap='Blues', xticklabels=class_names, yticklabels=class_names)

    plt.xticks(rotation=45, ha="right")
    plt.xlabel('Prediction', fontsize=12)
    plt.ylabel('Ground Truth', fontsize=12)

    # Enlarge the font of class_names on the tick labels
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=14)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=14)
    # plt.title(f'Confusion Matrix {level}{title_suffix}')
    plt.tight_layout()
    cm_save_path = os.path.join(args.result_dir, 'confusion_matrix.png')
    plt.savefig(cm_save_path)
    plt.close()  # Close the figure to avoid overlap

    # ---------------------
    # Plot the subset confusion matrix (rows and columns 5 to 12)
    # ---------------------
    # Slice the confusion matrix and corresponding annotation data.
    cm_subset = cm[5:13, 5:13]
    annot_subset = annot_data[5:13, 5:13]
    subset_class_names = class_names[5:13]

    plt.figure(figsize=(15, 12))
    ax_subset = sns.heatmap(cm_subset, annot=annot_subset, fmt=fmt, cmap='Blues',
                            xticklabels=subset_class_names, yticklabels=subset_class_names)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel('Prediction', fontsize=12)
    plt.ylabel('Ground Truth', fontsize=12)
    ax_subset.set_xticklabels(ax_subset.get_xticklabels(), fontsize=14)
    ax_subset.set_yticklabels(ax_subset.get_yticklabels(), fontsize=14)
    plt.tight_layout()
    subcm_save_path = os.path.join(args.result_dir, 'confusion_matrix_subset.png')
    plt.savefig(subcm_save_path)
    plt.close()

def compare_cm(cm1_path, cm2_path, class_names: list, normalized: bool = False) -> None:
    def _custom_format(x):
        """
        Custom formatting for the confusion matrix.
        If an entry is smaller than 0.01 in absolute value, it returns '0';
        otherwise, it returns the value formatted to 2 decimal places.
        """
        if abs(x) < 0.01:
            return '0'
        else:
            return f'{x:.2f}'

    # Load confusion matrices
    cm1 = np.load(cm1_path)
    cm2 = np.load(cm2_path)

    # Compute the difference (delta) between the two CMs: Model2 - Model1.
    delta = cm2 - cm1

    # Prepare annotations.
    # For the first CM, we keep the original annotations.
    if normalized:
        annot1 = np.array([[_custom_format(val) for val in row] for row in cm1])
    else:
        annot1 = cm1.astype(int).astype(str)

    # For the second CM, annotate each cell with its original value from Model2 and
    # the change (delta) from Model1. For instance: "v2\n(Δ: diff)".
    annot2 = []
    if normalized:
        for i in range(cm2.shape[0]):
            row = []
            for j in range(cm2.shape[1]):
                cell_val = _custom_format(cm2[i, j])
                diff = _custom_format(delta[i, j])
                # Add a plus sign if the difference is positive.
                sign = '+' if delta[i, j] > 0 else ''
                row.append(f"{cell_val}\n({sign}{diff})")
            annot2.append(row)
    else:
        for i in range(cm2.shape[0]):
            row = []
            for j in range(cm2.shape[1]):
                cell_val = str(cm2[i, j])
                diff = delta[i, j]
                sign = '+' if diff > 0 else ''
                row.append(f"{cell_val}\n({sign}{diff})")
            annot2.append(row)
    annot2 = np.array(annot2)

    # subset
    cm1 = cm1[5:13, 5:13]
    cm2 = cm2[5:13, 5:13]
    annot1 = annot1[5:13, 5:13]
    annot2 = annot2[5:13, 5:13]
    subset_class_names = class_names[5:13]

    # Set up a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(30, 12), sharey=True)

    # ----------------------
    # Plot for Model 1 (unchanged CM)
    # ----------------------
    sns.heatmap(cm1, annot=annot1, fmt='' if normalized else 'd', cmap='Blues',
                xticklabels=subset_class_names, yticklabels=subset_class_names, ax=axes[0],
                annot_kws={'fontsize': 18})
    axes[0].set_title("Sup CM", fontsize=22)
    axes[0].set_xlabel("Predicted", fontsize=18)
    axes[0].set_ylabel("Ground Truth", fontsize=18)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), fontsize=20, rotation=45, ha="right")
    axes[0].set_yticklabels(axes[0].get_yticklabels(), fontsize=20)

    # ----------------------
    # Plot for Model 2 (CM with delta)
    # ----------------------
    sns.heatmap(cm2, annot=annot2, fmt='', cmap='Blues',
                xticklabels=subset_class_names, yticklabels=subset_class_names, ax=axes[1],
                annot_kws={'fontsize': 18})
    axes[1].set_title("SupCon CM", fontsize=22)
    axes[1].set_xlabel("Predicted", fontsize=18)
    axes[1].set_ylabel("")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), fontsize=20, rotation=45, ha="right")
    # axes[1].set_yticklabels(axes[1].get_yticklabels(), fontsize=16)

    plt.tight_layout()

    # Save or show the figure. For example, saving to file:
    save_path = os.path.join(args.result_dir, 'confusion_matrix_comparison.png')
    plt.savefig(save_path)
    plt.close()

def cluster_quality(result_dir: str):
    """
    This function calculates the Calinski-Harabasz Index and Davies–Bouldin Index to evaluate the clustering quality of embeddings.
    :param result_dir:
    :return:
    """
    emb = pd.read_csv(os.path.join(result_dir, 'umap_results.csv'))
    X = emb[['umap_x', 'umap_y']].values
    labels = emb['label'].values

    # Calculate Calinski-Harabasz Index
    ch_index = calinski_harabasz_score(X, labels)
    print(f'{result_dir} Overall Calinski-Harabasz Index: {ch_index}')

    # Calculate Davies-Bouldin Index
    db_index = davies_bouldin_score(X, labels)
    print(f'{result_dir} Overall Davies-Bouldin Index: {db_index}')

    # Calculate CHI and DBI for subsets
    hoi_dict = {'grassland': [5, 6, 7, 8, 9], 'wetland': [11, 12], 'woodland': [1, 2]}
    habitats_of_interest = [5, 6, 7, 8, 9]
    for hab, label_list in hoi_dict.items():
        emb_subset = emb[emb['label'].isin(label_list)]
        X_subset = emb_subset[['umap_x', 'umap_y']].values
        labels_subset = emb_subset['label'].values

        ch_index = calinski_harabasz_score(X_subset, labels_subset)
        db_index = davies_bouldin_score(X_subset, labels_subset)
        print(f'Calinski-Harabasz Index ({hab}): {ch_index}')
        print(f'Davies-Bouldin Index ({hab}): {db_index}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate performance metrics between correctly classified and misclassified samples."
    )

    parser.add_argument(
        "--result-dir",
        type=str,
        default="data/cs_20192023_swintbase_img384",
        help="Path to the folder that contains the CSV files with classification results."
    )

    parser.add_argument(
        "--compared-dir",
        type=str,
        default="data/cs_20192023_swint_supcon_pretrain_baseline",
        help="Path to the folder that contains the CSV files with classification results."
    )

    args = parser.parse_args()
    main_result_dir = args.result_dir
    l2_metrics_df, kappa_score = performance_calculator(main_result_dir)
    # tools.display_dataframe_to_user(name="L2 Classification Metrics", dataframe=l2_metrics_df)
    print(l2_metrics_df)
    print(f'The kappa score is {kappa_score}.')

    # # Draw the CM for the main experiment
    # class_names = list(REASSIGN_LABEL_NAME_L3.values())
    # main_cm_path = os.path.join(main_result_dir, 'confusion_matrix_l3_normalized_cvtest.npy')
    # # plot_cm(main_cm_path, class_names, normalized=True)
    #
    # # Compare the main CM with a second CM
    # compared_dir = args.compared_dir
    # compared_cm_path = os.path.join(compared_dir, 'confusion_matrix_l3_normalized_cvtest.npy')
    # compare_cm(main_cm_path, compared_cm_path, class_names, normalized=True)

    # Evaluate the clustering performance
    # cluster_quality(main_result_dir)
