"""
Evaluation module.

This module implements functions for evaluating and analyzing the performance
of the insurance taxonomy classifier.
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Set, Optional
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

# Configure logging
logger = logging.getLogger(__name__)


def analyze_label_distribution(labels_list: List[List[str]]) -> pd.DataFrame:
    """
    Analyze the distribution of labels in the classification results.
    
    Args:
        labels_list: List of assigned label lists for each company
        
    Returns:
        DataFrame with label distribution statistics
    """
    # Flatten the list of lists to count all labels
    all_labels = [label for sublist in labels_list for label in sublist]
    
    # Count occurrences of each label
    label_counts = Counter(all_labels)
    
    # Convert to DataFrame
    distribution_df = pd.DataFrame.from_dict(label_counts, orient='index', 
                                            columns=['count']).reset_index()
    distribution_df.columns = ['label', 'count']
    
    # Sort by count in descending order
    distribution_df = distribution_df.sort_values('count', ascending=False).reset_index(drop=True)
    
    # Calculate percentage
    total_labels = sum(label_counts.values())
    distribution_df['percentage'] = (distribution_df['count'] / total_labels * 100).round(2)
    
    # Add cumulative percentage
    distribution_df['cumulative_percentage'] = distribution_df['percentage'].cumsum().round(2)
    
    return distribution_df


def analyze_coverage_by_attribute(company_df: pd.DataFrame, 
                                attribute_column: str) -> pd.DataFrame:
    """
    Analyze label coverage grouped by a company attribute.
    
    Args:
        company_df: DataFrame with company data and assigned labels
        attribute_column: Column to group by (e.g., 'sector', 'category')
        
    Returns:
        DataFrame with coverage statistics by attribute
    """
    # Group by the attribute column
    grouped = company_df.groupby(attribute_column)
    
    # Calculate statistics
    stats = []
    
    for group_name, group_df in grouped:
        # Count number of companies in this group
        group_count = len(group_df)
        
        # Count companies with at least one label
        has_label = sum(group_df['insurance_label'].str.len() > 0)
        
        # Calculate average number of labels
        avg_labels = group_df['insurance_label'].apply(
            lambda x: len(x.split('; ')) if isinstance(x, str) and x else 0
        ).mean()
        
        # Add to statistics
        stats.append({
            attribute_column: group_name,
            'company_count': group_count,
            'with_label_count': has_label,
            'coverage_percentage': (has_label / group_count * 100).round(2),
            'avg_labels': round(avg_labels, 2)
        })
    
    # Convert to DataFrame and sort by coverage
    coverage_df = pd.DataFrame(stats)
    coverage_df = coverage_df.sort_values('coverage_percentage', ascending=False).reset_index(drop=True)
    
    return coverage_df


def visualize_label_distribution(distribution_df: pd.DataFrame, 
                               top_n: int = 20,
                               figsize: Tuple[int, int] = (12, 8)):
    """
    Visualize the distribution of labels.
    
    Args:
        distribution_df: DataFrame with label distribution statistics
        top_n: Number of top labels to show
        figsize: Figure size
    """
    # Take top N labels
    plot_df = distribution_df.head(top_n)
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot bar chart
    ax = sns.barplot(x='label', y='count', data=plot_df)
    
    # Add percentage text on bars
    for i, row in plot_df.iterrows():
        ax.text(i, row['count'] * 0.5, f"{row['percentage']}%", 
                ha='center', va='center', color='white', fontweight='bold')
    
    # Rotate x labels
    plt.xticks(rotation=45, ha='right')
    
    # Add labels and title
    plt.xlabel('Insurance Label')
    plt.ylabel('Count')
    plt.title(f'Top {top_n} Insurance Labels Distribution')
    
    # Adjust layout
    plt.tight_layout()
    
    return ax


def visualize_coverage_by_attribute(coverage_df: pd.DataFrame, 
                                   attribute_column: str,
                                   top_n: int = 15,
                                   figsize: Tuple[int, int] = (12, 8)):
    """
    Visualize the coverage by company attribute.
    
    Args:
        coverage_df: DataFrame with coverage statistics by attribute
        attribute_column: Column used for grouping
        top_n: Number of top attributes to show
        figsize: Figure size
    """
    # Take top N attributes
    plot_df = coverage_df.head(top_n)
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot bar chart
    ax = sns.barplot(x=attribute_column, y='coverage_percentage', data=plot_df)
    
    # Add percentage text on bars
    for i, row in plot_df.iterrows():
        ax.text(i, row['coverage_percentage'] * 0.5, f"{row['coverage_percentage']}%", 
                ha='center', va='center', color='white', fontweight='bold')
    
    # Rotate x labels
    plt.xticks(rotation=45, ha='right')
    
    # Add labels and title
    plt.xlabel(attribute_column.capitalize())
    plt.ylabel('Coverage Percentage')
    plt.title(f'Insurance Label Coverage by {attribute_column.capitalize()}')
    
    # Adjust layout
    plt.tight_layout()
    
    return ax


def analyze_misclassifications(prediction_df: pd.DataFrame,
                              ground_truth_column: str,
                              prediction_column: str = 'insurance_label',
                              delimiter: str = '; ') -> Dict:
    """
    Analyze misclassifications if ground truth is available.
    
    Args:
        prediction_df: DataFrame with predictions
        ground_truth_column: Column with ground truth labels
        prediction_column: Column with predicted labels
        delimiter: Delimiter used in label strings
        
    Returns:
        Dictionary with misclassification statistics
    """
    # Create sets of labels for easier comparison
    prediction_df['ground_truth_set'] = prediction_df[ground_truth_column].apply(
        lambda x: set(x.split(delimiter)) if isinstance(x, str) and x else set()
    )
    
    prediction_df['prediction_set'] = prediction_df[prediction_column].apply(
        lambda x: set(x.split(delimiter)) if isinstance(x, str) and x else set()
    )
    
    # Calculate overlap metrics
    prediction_df['true_positives'] = prediction_df.apply(
        lambda row: len(row['ground_truth_set'] & row['prediction_set']), axis=1
    )
    
    prediction_df['false_positives'] = prediction_df.apply(
        lambda row: len(row['prediction_set'] - row['ground_truth_set']), axis=1
    )
    
    prediction_df['false_negatives'] = prediction_df.apply(
        lambda row: len(row['ground_truth_set'] - row['prediction_set']), axis=1
    )
    
    # Calculate precision and recall for each sample
    prediction_df['precision'] = prediction_df.apply(
        lambda row: row['true_positives'] / max(1, len(row['prediction_set'])), axis=1
    )
    
    prediction_df['recall'] = prediction_df.apply(
        lambda row: row['true_positives'] / max(1, len(row['ground_truth_set'])), axis=1
    )
    
    # Calculate F1 score
    prediction_df['f1_score'] = prediction_df.apply(
        lambda row: 2 * (row['precision'] * row['recall']) / 
                   max(0.001, row['precision'] + row['recall']), axis=1
    )
    
    # Calculate overall metrics
    metrics = {
        'average_precision': prediction_df['precision'].mean(),
        'average_recall': prediction_df['recall'].mean(),
        'average_f1_score': prediction_df['f1_score'].mean(),
        'perfect_matches': sum(prediction_df['ground_truth_set'] == prediction_df['prediction_set']),
        'perfect_match_percentage': (sum(prediction_df['ground_truth_set'] == prediction_df['prediction_set']) / 
                                    len(prediction_df) * 100),
        'samples_with_false_positives': sum(prediction_df['false_positives'] > 0),
        'samples_with_false_negatives': sum(prediction_df['false_negatives'] > 0)
    }
    
    return {
        'metrics': metrics,
        'detailed_df': prediction_df
    }


def generate_evaluation_report(company_df: pd.DataFrame,
                             prediction_results: Dict,
                             output_folder: str = 'results',
                             ground_truth_column: Optional[str] = None):
    """
    Generate a comprehensive evaluation report.
    
    Args:
        company_df: DataFrame with company data
        prediction_results: Results from the classifier's predict method
        output_folder: Folder to save report files
        ground_truth_column: Column with ground truth labels (if available)
    """
    # Ensure output folder exists
    import os
    os.makedirs(output_folder, exist_ok=True)
    
    # Create matched labels list for analysis
    matched_labels_list = prediction_results['matched_labels']
    
    # Analyze label distribution
    logger.info("Analyzing label distribution")
    distribution_df = analyze_label_distribution(matched_labels_list)
    distribution_df.to_csv(f"{output_folder}/label_distribution.csv", index=False)
    
    # Visualize label distribution
    logger.info("Visualizing label distribution")
    plt.figure(figsize=(12, 8))
    visualize_label_distribution(distribution_df)
    plt.savefig(f"{output_folder}/label_distribution.png", dpi=300, bbox_inches='tight')
    
    # Analyze coverage by sector if available
    if 'sector' in company_df.columns:
        logger.info("Analyzing coverage by sector")
        sector_coverage_df = analyze_coverage_by_attribute(company_df, 'sector')
        sector_coverage_df.to_csv(f"{output_folder}/sector_coverage.csv", index=False)
        
        # Visualize sector coverage
        logger.info("Visualizing sector coverage")
        plt.figure(figsize=(12, 8))
        visualize_coverage_by_attribute(sector_coverage_df, 'sector')
        plt.savefig(f"{output_folder}/sector_coverage.png", dpi=300, bbox_inches='tight')
    
    # Analyze coverage by category if available
    if 'category' in company_df.columns:
        logger.info("Analyzing coverage by category")
        category_coverage_df = analyze_coverage_by_attribute(company_df, 'category')
        category_coverage_df.to_csv(f"{output_folder}/category_coverage.csv", index=False)
        
        # Visualize category coverage
        logger.info("Visualizing category coverage")
        plt.figure(figsize=(12, 8))
        visualize_coverage_by_attribute(category_coverage_df, 'category')
        plt.savefig(f"{output_folder}/category_coverage.png", dpi=300, bbox_inches='tight')
    
    # Analyze misclassifications if ground truth is available
    if ground_truth_column and ground_truth_column in company_df.columns:
        logger.info("Analyzing misclassifications")
        misclassification_results = analyze_misclassifications(
            company_df, ground_truth_column)
        
        # Save metrics
        pd.DataFrame([misclassification_results['metrics']]).to_csv(
            f"{output_folder}/misclassification_metrics.csv", index=False)
        
        # Save detailed results
        misclassification_results['detailed_df'].to_csv(
            f"{output_folder}/detailed_misclassifications.csv", index=False)
    
    logger.info(f"Evaluation report generated in {output_folder}")


def cross_validate(classifier, embedding_generator, company_df, 
                  text_columns, n_splits=5, random_state=42):
    """
    Perform cross-validation to evaluate classifier performance.
    
    Args:
        classifier: Classifier to evaluate
        embedding_generator: Embedding generator instance
        company_df: DataFrame with company data
        text_columns: Columns to use for classification
        n_splits: Number of cross-validation splits
        random_state: Random state for reproducibility
        
    Returns:
        Dictionary with cross-validation results
    """
    from sklearn.model_selection import KFold
    
    # Initialize K-Fold cross-validator
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Store metrics for each fold
    fold_metrics = []
    
    # Perform cross-validation
    for fold, (train_idx, test_idx) in enumerate(kf.split(company_df)):
        logger.info(f"Evaluating fold {fold+1}/{n_splits}")
        
        # Split data
        train_df = company_df.iloc[train_idx]
        test_df = company_df.iloc[test_idx]
        
        # Combine text columns for training data
        train_texts = embedding_generator.combine_text_fields(train_df, text_columns)
        
        # Generate train embeddings
        train_embeddings = embedding_generator.generate_embeddings(train_texts)
        
        # Combine text columns for taxonomy
        taxonomy_texts = [label for label in classifier.taxonomy_labels]
        
        # Generate taxonomy embeddings
        taxonomy_embeddings = embedding_generator.generate_embeddings(taxonomy_texts)
        
        # Fit classifier
        classifier.fit(train_embeddings, taxonomy_embeddings, classifier.taxonomy_labels)
        
        # Evaluate on test data
        evaluation_results = evaluate_classifier(
            classifier, test_df, embedding_generator, text_columns)
        
        # Store metrics
        fold_metrics.append(evaluation_results['metrics'])
        
        # Log metrics for this fold
        logger.info(f"Fold {fold+1} metrics: {evaluation_results['metrics']}")
    
    # Calculate average metrics across folds
    avg_metrics = {}
    
    for key in fold_metrics[0].keys():
        avg_metrics[key] = sum(fold[key] for fold in fold_metrics) / n_splits
    
    # Add fold metrics to results
    results = {
        'fold_metrics': fold_metrics,
        'average_metrics': avg_metrics
    }
    
    logger.info(f"Cross-validation results: {avg_metrics}")
    
    return results