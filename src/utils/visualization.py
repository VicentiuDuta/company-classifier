"""
Visualization module.

This module implements visualization functions for the insurance taxonomy classifier.
"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
from collections import Counter

# Configure logging
logger = logging.getLogger(__name__)


def plot_embedding_clusters(embeddings: np.ndarray, 
                          labels: List[str],
                          title: str = "Embedding Clusters",
                          method: str = "tsne",
                          n_components: int = 2,
                          figsize: Tuple[int, int] = (12, 10),
                          output_path: Optional[str] = None,
                          show_labels: bool = True,
                          max_labels: int = 100,
                          random_state: int = 42):
    """
    Plot embeddings in 2D or 3D space using dimensionality reduction.
    
    Args:
        embeddings: Array of embeddings
        labels: List of labels for each embedding
        title: Plot title
        method: Dimensionality reduction method ('tsne' or 'pca')
        n_components: Number of components (2 or 3)
        figsize: Figure size
        output_path: Path to save the plot (if None, just display)
        show_labels: Whether to show text labels on points
        max_labels: Maximum number of labels to display
        random_state: Random state for reproducibility
    """
    if n_components not in [2, 3]:
        raise ValueError("n_components must be 2 or 3")
    
    # Apply dimensionality reduction
    if method.lower() == "tsne":
        logger.info(f"Applying t-SNE with {n_components} components")
        reducer = TSNE(n_components=n_components, random_state=random_state)
    elif method.lower() == "pca":
        logger.info(f"Applying PCA with {n_components} components")
        reducer = PCA(n_components=n_components, random_state=random_state)
    else:
        raise ValueError("Method must be 'tsne' or 'pca'")
    
    reduced_data = reducer.fit_transform(embeddings)
    
    # Create plot
    plt.figure(figsize=figsize)
    
    if n_components == 2:
        # 2D plot
        scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], 
                             c=np.arange(len(reduced_data)), cmap='viridis', alpha=0.8)
        
        # Add labels if requested
        if show_labels and len(labels) <= max_labels:
            for i, label in enumerate(labels):
                plt.annotate(label, (reduced_data[i, 0], reduced_data[i, 1]), 
                            fontsize=8, ha='right', va='bottom')
                
        # Set labels
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        
    else:
        # 3D plot
        ax = plt.axes(projection='3d')
        scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2],
                           c=np.arange(len(reduced_data)), cmap='viridis', alpha=0.8)
        
        # Add labels if requested
        if show_labels and len(labels) <= max_labels:
            for i, label in enumerate(labels):
                ax.text(reduced_data[i, 0], reduced_data[i, 1], reduced_data[i, 2], 
                       label, fontsize=8)
                
        # Set labels
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
    
    # Set title
    plt.title(title)
    
    # Add colorbar
    plt.colorbar(scatter, label='Index')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {output_path}")
    
    # Display
    plt.show()


def plot_similarity_heatmap(similarity_matrix: np.ndarray,
                          x_labels: List[str],
                          y_labels: List[str],
                          title: str = "Similarity Heatmap",
                          figsize: Tuple[int, int] = (12, 10),
                          output_path: Optional[str] = None,
                          cmap: str = "YlGnBu",
                          max_labels: int = 30):
    """
    Plot a heatmap of similarity scores.
    
    Args:
        similarity_matrix: Matrix of similarity scores
        x_labels: Labels for the x-axis
        y_labels: Labels for the y-axis
        title: Plot title
        figsize: Figure size
        output_path: Path to save the plot (if None, just display)
        cmap: Colormap to use
        max_labels: Maximum number of labels to display
    """
    # Limit the number of labels to display
    if len(x_labels) > max_labels or len(y_labels) > max_labels:
        logger.info(f"Limiting heatmap to {max_labels} labels")
        similarity_matrix = similarity_matrix[:max_labels, :max_labels]
        x_labels = x_labels[:max_labels]
        y_labels = y_labels[:max_labels]
    
    # Create plot
    plt.figure(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(similarity_matrix, annot=True, fmt=".2f", cmap=cmap,
               xticklabels=x_labels, yticklabels=y_labels)
    
    # Set title
    plt.title(title)
    
    # Rotate x labels
    plt.xticks(rotation=45, ha='right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Heatmap saved to {output_path}")
    
    # Display
    plt.show()


def plot_label_counts(labels_list: List[List[str]],
                     top_n: int = 20,
                     title: str = "Top Insurance Labels",
                     figsize: Tuple[int, int] = (12, 8),
                     output_path: Optional[str] = None):
    """
    Plot counts of the most frequent labels.
    
    Args:
        labels_list: List of assigned label lists for each company
        top_n: Number of top labels to show
        title: Plot title
        figsize: Figure size
        output_path: Path to save the plot (if None, just display)
    """
    # Flatten the list of lists to count all labels
    all_labels = [label for sublist in labels_list for label in sublist]
    
    # Count occurrences of each label
    label_counts = Counter(all_labels)
    
    # Get top N labels
    top_labels = label_counts.most_common(top_n)
    
    # Extract labels and counts
    labels = [label for label, _ in top_labels]
    counts = [count for _, count in top_labels]
    
    # Create plot
    plt.figure(figsize=figsize)
    
    # Plot horizontal bar chart
    bars = plt.barh(labels, counts, color='skyblue')
    
    # Add counts as text
    for i, bar in enumerate(bars):
        plt.text(bar.get_width() + 0.5, i, str(counts[i]), va='center')
    
    # Invert y-axis to show top labels first
    plt.gca().invert_yaxis()
    
    # Set labels
    plt.xlabel('Count')
    plt.ylabel('Insurance Label')
    
    # Set title
    plt.title(title)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {output_path}")
    
    # Display
    plt.show()


def plot_labels_per_company(labels_list: List[List[str]],
                          title: str = "Distribution of Labels per Company",
                          figsize: Tuple[int, int] = (10, 6),
                          output_path: Optional[str] = None):
    """
    Plot distribution of number of labels per company.
    
    Args:
        labels_list: List of assigned label lists for each company
        title: Plot title
        figsize: Figure size
        output_path: Path to save the plot (if None, just display)
    """
    # Count number of labels per company
    label_counts = [len(labels) for labels in labels_list]
    
    # Create plot
    plt.figure(figsize=figsize)
    
    # Plot histogram
    bins = range(max(label_counts) + 2)
    plt.hist(label_counts, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Calculate percentage of companies with no labels
    no_labels_percent = (label_counts.count(0) / len(label_counts)) * 100
    plt.text(0.5, 0.9, f"Companies with no labels: {no_labels_percent:.1f}%",
            transform=plt.gca().transAxes, fontsize=12, ha='center')
    
    # Set labels
    plt.xlabel('Number of Labels')
    plt.ylabel('Number of Companies')
    
    # Set title
    plt.title(title)
    
    # Set x-ticks to integers
    plt.xticks(range(max(label_counts) + 1))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {output_path}")
    
    # Display
    plt.show()


def plot_coverage_by_attribute(company_df: pd.DataFrame,
                             attribute_column: str,
                             insurance_label_column: str = 'insurance_label',
                             top_n: int = 15,
                             title: Optional[str] = None,
                             figsize: Tuple[int, int] = (12, 8),
                             output_path: Optional[str] = None):
    """
    Plot coverage by company attribute.
    
    Args:
        company_df: DataFrame with company data and assigned labels
        attribute_column: Column to group by (e.g., 'sector', 'category')
        insurance_label_column: Column with insurance labels
        top_n: Number of top attributes to show
        title: Plot title (if None, generate based on attribute)
        figsize: Figure size
        output_path: Path to save the plot (if None, just display)
    """
    # Generate title if not provided
    if title is None:
        title = f"Insurance Label Coverage by {attribute_column.capitalize()}"
    
    # Group by the attribute column
    grouped = company_df.groupby(attribute_column)
    
    # Calculate statistics
    stats = []
    
    for group_name, group_df in grouped:
        # Count number of companies in this group
        group_count = len(group_df)
        
        # Count companies with at least one label
        has_label = sum(group_df[insurance_label_column].str.len() > 0)
        
        # Calculate coverage percentage
        coverage = (has_label / group_count * 100)
        
        # Add to statistics
        stats.append({
            attribute_column: group_name,
            'company_count': group_count,
            'coverage_percentage': coverage
        })
    
    # Convert to DataFrame and sort by coverage
    coverage_df = pd.DataFrame(stats)
    coverage_df = coverage_df.sort_values('coverage_percentage', ascending=False).head(top_n)
    
    # Create plot
    plt.figure(figsize=figsize)
    
    # Plot bar chart
    bars = plt.bar(coverage_df[attribute_column], coverage_df['coverage_percentage'], color='skyblue')
    
    # Add percentage text on bars
    for i, bar in enumerate(bars):
        plt.text(i, bar.get_height() * 0.5, f"{coverage_df['coverage_percentage'].iloc[i]:.1f}%",
                ha='center', va='center', color='black', fontweight='bold')
    
    # Rotate x labels
    plt.xticks(rotation=45, ha='right')
    
    # Set labels
    plt.xlabel(attribute_column.capitalize())
    plt.ylabel('Coverage Percentage')
    
    # Set title
    plt.title(title)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {output_path}")
    
    # Display
    plt.show()


def generate_report_dashboard(company_df: pd.DataFrame,
                             prediction_results: Dict,
                             output_folder: str = 'results/dashboard',
                             taxonomy_labels: Optional[List[str]] = None):
    """
    Generate a comprehensive dashboard of visualizations.
    
    Args:
        company_df: DataFrame with company data
        prediction_results: Results from the classifier's predict method
        output_folder: Folder to save dashboard files
        taxonomy_labels: List of taxonomy labels (if available)
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Extract matched labels list
    matched_labels_list = prediction_results['matched_labels']
    
    logger.info("Generating dashboard visualizations")
    
    # Plot label counts
    logger.info("Plotting label counts")
    plot_label_counts(
        matched_labels_list, 
        output_path=f"{output_folder}/top_labels.png"
    )
    
    # Plot labels per company
    logger.info("Plotting labels per company distribution")
    plot_labels_per_company(
        matched_labels_list,
        output_path=f"{output_folder}/labels_per_company.png"
    )
    
    # Plot coverage by sector if available
    if 'sector' in company_df.columns:
        logger.info("Plotting coverage by sector")
        plot_coverage_by_attribute(
            company_df, 
            'sector',
            output_path=f"{output_folder}/sector_coverage.png"
        )
    
    # Plot coverage by category if available
    if 'category' in company_df.columns:
        logger.info("Plotting coverage by category")
        plot_coverage_by_attribute(
            company_df, 
            'category',
            output_path=f"{output_folder}/category_coverage.png"
        )
    
    # Plot taxonomy embedding clusters if available
    if taxonomy_labels and 'taxonomy_embeddings' in prediction_results:
        logger.info("Plotting taxonomy embedding clusters")
        plot_embedding_clusters(
            prediction_results['taxonomy_embeddings'], 
            taxonomy_labels,
            title="Insurance Taxonomy Embeddings",
            output_path=f"{output_folder}/taxonomy_clusters.png"
        )
    
    logger.info(f"Dashboard generated in {output_folder}")