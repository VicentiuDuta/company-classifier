"""
Clustering module.

This module implements clustering algorithms for grouping similar companies
and mapping clusters to taxonomy labels.
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os

# Configure logging
logger = logging.getLogger(__name__)


class CompanyClustering:
    """Class for clustering companies based on their embeddings."""
    
    def __init__(self, 
                 algorithm: str = "kmeans",
                 n_clusters: int = 50,
                 min_cluster_size: int = 5,
                 random_state: int = 42):
        """
        Initialize the clustering model.
        
        Args:
            algorithm: Clustering algorithm to use ('kmeans' or 'dbscan')
            n_clusters: Number of clusters for KMeans
            min_cluster_size: Minimum cluster size for DBSCAN
            random_state: Random state for reproducibility
        """
        self.algorithm = algorithm.lower()
        self.n_clusters = n_clusters
        self.min_cluster_size = min_cluster_size
        self.random_state = random_state
        self.model = None
        self.cluster_labels = None
        self.cluster_centers = None
        
    def fit(self, embeddings: np.ndarray):
        """
        Fit the clustering model.
        
        Args:
            embeddings: Array of company embeddings
        """
        logger.info(f"Fitting {self.algorithm} clustering with {len(embeddings)} embeddings")
        
        if self.algorithm == "kmeans":
            self.model = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init="auto"
            )
            self.cluster_labels = self.model.fit_predict(embeddings)
            self.cluster_centers = self.model.cluster_centers_
            
        elif self.algorithm == "dbscan":
            # Find optimal eps parameter using k-distance graph
            eps = self._estimate_eps(embeddings)
            
            self.model = DBSCAN(
                eps=eps,
                min_samples=self.min_cluster_size
            )
            self.cluster_labels = self.model.fit_predict(embeddings)
            
            # Extract cluster centers by averaging embeddings in each cluster
            unique_clusters = np.unique(self.cluster_labels)
            unique_clusters = unique_clusters[unique_clusters != -1]  # Remove noise cluster
            
            self.cluster_centers = np.zeros((len(unique_clusters), embeddings.shape[1]))
            
            for i, cluster_id in enumerate(unique_clusters):
                cluster_mask = self.cluster_labels == cluster_id
                self.cluster_centers[i] = embeddings[cluster_mask].mean(axis=0)
        
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        
        # Evaluate clustering
        self._evaluate_clustering(embeddings)
        
    def _estimate_eps(self, embeddings: np.ndarray, k: int = 5) -> float:
        """
        Estimate optimal eps parameter for DBSCAN.
        
        Args:
            embeddings: Array of embeddings
            k: Number of nearest neighbors to consider
            
        Returns:
            Estimated eps value
        """
        from sklearn.neighbors import NearestNeighbors
        
        # Fit nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=k).fit(embeddings)
        distances, _ = nbrs.kneighbors(embeddings)
        
        # Sort distances to kth nearest neighbor
        distances = np.sort(distances[:, k-1])
        
        # Find elbow point using kneedle algorithm
        try:
            from kneed import KneeLocator
            kneedle = KneeLocator(
                range(len(distances)), 
                distances, 
                S=1.0, 
                curve="convex", 
                direction="increasing"
            )
            eps = distances[kneedle.knee]
        except (ImportError, Exception):
            # If kneedle not available or fails, use heuristic
            eps = np.percentile(distances, 90)
        
        logger.info(f"Estimated DBSCAN eps: {eps:.4f}")
        return eps
    
    def _evaluate_clustering(self, embeddings: np.ndarray):
        """
        Evaluate clustering quality.
        
        Args:
            embeddings: Array of embeddings
        """
        # Count clusters
        if self.algorithm == "kmeans":
            n_clusters = self.n_clusters
        else:
            unique_clusters = np.unique(self.cluster_labels)
            n_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
        
        # Count samples in each cluster
        cluster_counts = Counter(self.cluster_labels)
        
        # Remove noise cluster (-1) for statistics
        if -1 in cluster_counts:
            noise_count = cluster_counts.pop(-1)
            noise_percentage = noise_count / len(self.cluster_labels) * 100
        else:
            noise_count = 0
            noise_percentage = 0
        
        # Calculate cluster sizes
        min_size = min(cluster_counts.values()) if cluster_counts else 0
        max_size = max(cluster_counts.values()) if cluster_counts else 0
        avg_size = sum(cluster_counts.values()) / len(cluster_counts) if cluster_counts else 0
        
        # Calculate silhouette score if more than one cluster
        if n_clusters > 1:
            try:
                # For DBSCAN, exclude noise points
                if self.algorithm == "dbscan" and -1 in np.unique(self.cluster_labels):
                    mask = self.cluster_labels != -1
                    silhouette = silhouette_score(
                        embeddings[mask], self.cluster_labels[mask])
                else:
                    silhouette = silhouette_score(embeddings, self.cluster_labels)
            except Exception as e:
                logger.warning(f"Failed to calculate silhouette score: {e}")
                silhouette = None
        else:
            silhouette = None
        
        # Log statistics
        logger.info(f"Clustering results: {n_clusters} clusters formed")
        logger.info(f"Cluster sizes - Min: {min_size}, Max: {max_size}, Avg: {avg_size:.1f}")
        
        if noise_count > 0:
            logger.info(f"Noise points: {noise_count} ({noise_percentage:.1f}%)")
            
        if silhouette is not None:
            logger.info(f"Silhouette score: {silhouette:.4f}")
    
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new embeddings.
        
        Args:
            embeddings: Array of embeddings
            
        Returns:
            Array of cluster labels
        """
        if self.model is None:
            raise ValueError("Clustering model not fitted yet")
        
        if self.algorithm == "kmeans":
            return self.model.predict(embeddings)
        elif self.algorithm == "dbscan":
            # For DBSCAN, we need to assign to nearest cluster center
            return self._assign_to_nearest_cluster(embeddings)
    
    def _assign_to_nearest_cluster(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Assign embeddings to nearest cluster center.
        
        Args:
            embeddings: Array of embeddings
            
        Returns:
            Array of cluster labels
        """
        # Calculate distances to cluster centers
        distances = np.zeros((len(embeddings), len(self.cluster_centers)))
        
        for i, center in enumerate(self.cluster_centers):
            # Calculate Euclidean distance
            diff = embeddings - center
            distances[:, i] = np.sqrt(np.sum(diff * diff, axis=1))
        
        # Assign to nearest cluster
        return np.argmin(distances, axis=1)
    
    def map_clusters_to_taxonomy(self, 
                              taxonomy_embeddings: np.ndarray,
                              taxonomy_labels: List[str],
                              top_k: int = 3) -> Dict[int, List[Tuple[str, float]]]:
        """
        Map clusters to taxonomy labels based on similarity.
        
        Args:
            taxonomy_embeddings: Array of taxonomy embeddings
            taxonomy_labels: List of taxonomy labels
            top_k: Number of top labels to assign to each cluster
            
        Returns:
            Dictionary mapping cluster IDs to lists of (label, similarity) tuples
        """
        if self.cluster_centers is None:
            raise ValueError("Clustering model not fitted yet")
        
        # Normalize embeddings
        cluster_norm = self.cluster_centers / np.linalg.norm(self.cluster_centers, axis=1, keepdims=True)
        taxonomy_norm = taxonomy_embeddings / np.linalg.norm(taxonomy_embeddings, axis=1, keepdims=True)
        
        # Calculate similarity matrix
        similarity_matrix = np.dot(cluster_norm, taxonomy_norm.T)
        
        # Map clusters to taxonomy labels
        cluster_to_labels = {}
        
        for cluster_id in range(len(self.cluster_centers)):
            # Get similarities for this cluster
            similarities = similarity_matrix[cluster_id]
            
            # Get top k similar labels
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            # Map to (label, similarity) tuples
            cluster_to_labels[cluster_id] = [
                (taxonomy_labels[idx], similarities[idx]) 
                for idx in top_indices
            ]
        
        return cluster_to_labels
    
    def get_cluster_samples(self, 
                          cluster_id: int,
                          company_df: pd.DataFrame) -> pd.DataFrame:
        """
        Get samples belonging to a specific cluster.
        
        Args:
            cluster_id: Cluster ID
            company_df: DataFrame with company data
            
        Returns:
            DataFrame with companies in the cluster
        """
        if self.cluster_labels is None:
            raise ValueError("Clustering model not fitted yet")
        
        # Find indices of companies in the cluster
        cluster_indices = np.where(self.cluster_labels == cluster_id)[0]
        
        # Return subset of DataFrame
        return company_df.iloc[cluster_indices].copy()
    
    def assign_cluster_labels(self, 
                           company_df: pd.DataFrame,
                           cluster_to_labels: Dict[int, List[Tuple[str, float]]],
                           similarity_threshold: float = 0.5,
                           delimiter: str = "; ") -> pd.DataFrame:
        """
        Assign taxonomy labels to companies based on their clusters.
        
        Args:
            company_df: DataFrame with company data
            cluster_to_labels: Mapping from cluster IDs to taxonomy labels
            similarity_threshold: Minimum similarity score to assign a label
            delimiter: Delimiter for joining multiple labels
            
        Returns:
            DataFrame with assigned insurance labels
        """
        if self.cluster_labels is None:
            raise ValueError("Clustering model not fitted yet")
        
        # Create copy of the DataFrame
        result_df = company_df.copy()
        
        # Add cluster column
        result_df["cluster_id"] = self.cluster_labels
        
        # Initialize insurance_label column
        result_df["insurance_label"] = ""
        
        # Assign labels based on clusters
        for i, cluster_id in enumerate(self.cluster_labels):
            # Skip noise cluster
            if cluster_id == -1:
                continue
                
            # Get labels for this cluster
            cluster_labels = cluster_to_labels.get(cluster_id, [])
            
            # Filter by similarity threshold
            filtered_labels = [
                label for label, similarity in cluster_labels
                if similarity >= similarity_threshold
            ]
            
            # Join labels
            result_df.iloc[i, result_df.columns.get_loc("insurance_label")] = delimiter.join(filtered_labels)
        
        return result_df
    
    def visualize_clusters(self, 
                         embeddings: np.ndarray,
                         method: str = "tsne",
                         output_path: Optional[str] = None):
        """
        Visualize clusters using dimensionality reduction.
        
        Args:
            embeddings: Array of embeddings
            method: Dimensionality reduction method ('tsne' or 'pca')
            output_path: Path to save the visualization
        """
        if self.cluster_labels is None:
            raise ValueError("Clustering model not fitted yet")
        
        # Apply dimensionality reduction
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        
        if method.lower() == "tsne":
            reducer = TSNE(n_components=2, random_state=self.random_state)
        else:
            reducer = PCA(n_components=2, random_state=self.random_state)
        
        reduced_data = reducer.fit_transform(embeddings)
        
        # Create scatter plot
        plt.figure(figsize=(12, 10))
        
        # Use different colors for each cluster
        unique_clusters = np.unique(self.cluster_labels)
        
        # Create color map (excluding noise)
        if -1 in unique_clusters:
            colormap = plt.cm.get_cmap("viridis", len(unique_clusters) - 1)
        else:
            colormap = plt.cm.get_cmap("viridis", len(unique_clusters))
        
        # Plot each cluster
        for i, cluster_id in enumerate(unique_clusters):
            # Special handling for noise cluster
            if cluster_id == -1:
                cluster_mask = self.cluster_labels == -1
                plt.scatter(
                    reduced_data[cluster_mask, 0],
                    reduced_data[cluster_mask, 1],
                    s=10, c="black", marker="x", alpha=0.3,
                    label=f"Noise ({np.sum(cluster_mask)})"
                )
            else:
                cluster_mask = self.cluster_labels == cluster_id
                plt.scatter(
                    reduced_data[cluster_mask, 0],
                    reduced_data[cluster_mask, 1],
                    s=30, c=[colormap(i)], alpha=0.7,
                    label=f"Cluster {cluster_id} ({np.sum(cluster_mask)})"
                )
        
        # Plot cluster centers
        if self.algorithm == "kmeans":
            # Reduce cluster centers
            centers_reduced = reducer.transform(self.cluster_centers)
            plt.scatter(
                centers_reduced[:, 0],
                centers_reduced[:, 1],
                s=100, c="red", marker="*",
                label="Cluster Centers"
            )
        
        # Add labels and legend
        plt.title(f"Cluster Visualization ({method.upper()})")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        
        # Create custom legend with limited entries
        if len(unique_clusters) > 15:
            # Show first few clusters, noise, and centers
            handles, labels = plt.gca().get_legend_handles_labels()
            selected_indices = list(range(min(10, len(handles) - 2))) + [-2, -1]
            plt.legend([handles[i] for i in selected_indices],
                      [labels[i] for i in selected_indices],
                      loc="best", fontsize=8)
        else:
            plt.legend(loc="best", fontsize=8)
        
        # Save if output path provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Cluster visualization saved to {output_path}")
        
        plt.tight_layout()
        plt.show()
    
    def visualize_cluster_sizes(self, output_path: Optional[str] = None):
        """
        Visualize the distribution of cluster sizes.
        
        Args:
            output_path: Path to save the visualization
        """
        if self.cluster_labels is None:
            raise ValueError("Clustering model not fitted yet")
        
        # Count samples in each cluster
        cluster_counts = Counter(self.cluster_labels)
        
        # Remove noise cluster for visualization
        if -1 in cluster_counts:
            noise_count = cluster_counts.pop(-1)
        else:
            noise_count = 0
        
        # Sort clusters by size
        sorted_clusters = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)
        cluster_ids = [str(cluster_id) for cluster_id, _ in sorted_clusters]
        sizes = [count for _, count in sorted_clusters]
        
        # Create bar plot
        plt.figure(figsize=(12, 6))
        
        # Plot top clusters
        max_clusters_to_show = 50
        if len(cluster_ids) > max_clusters_to_show:
            visible_cluster_ids = cluster_ids[:max_clusters_to_show]
            visible_sizes = sizes[:max_clusters_to_show]
        else:
            visible_cluster_ids = cluster_ids
            visible_sizes = sizes
            
        bars = plt.bar(visible_cluster_ids, visible_sizes, color='skyblue')
        
        # Add noise cluster if present
        if noise_count > 0:
            plt.text(0.02, 0.95, f"Noise points: {noise_count}", 
                    transform=plt.gca().transAxes, fontsize=10)
        
        # Add labels
        plt.xlabel('Cluster ID')
        plt.ylabel('Number of Companies')
        plt.title('Cluster Size Distribution')
        
        # Rotate x labels if many clusters
        if len(visible_cluster_ids) > 10:
            plt.xticks(rotation=90)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if output path provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Cluster size visualization saved to {output_path}")
        
        plt.show()
        
    def save(self, filepath: str):
        """
        Save the clustering model to a file.
        
        Args:
            filepath: Path to save the model to
        """
        import pickle
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'algorithm': self.algorithm,
                'n_clusters': self.n_clusters,
                'min_cluster_size': self.min_cluster_size,
                'random_state': self.random_state,
                'cluster_labels': self.cluster_labels,
                'cluster_centers': self.cluster_centers,
                'model': self.model
            }, f)
        
        logger.info(f"Clustering model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """
        Load a clustering model from a file.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Loaded clustering model
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        # Create a new instance
        clustering = cls(
            algorithm=data['algorithm'],
            n_clusters=data['n_clusters'],
            min_cluster_size=data['min_cluster_size'],
            random_state=data['random_state']
        )
        
        # Set attributes
        clustering.model = data['model']
        clustering.cluster_labels = data['cluster_labels']
        clustering.cluster_centers = data['cluster_centers']
        
        logger.info(f"Clustering model loaded from {filepath}")
        return clustering


def optimize_clusters(embeddings: np.ndarray, 
                    min_clusters: int = 10, 
                    max_clusters: int = 100,
                    step: int = 10,
                    random_state: int = 42) -> Tuple[int, float]:
    """
    Find optimal number of clusters using silhouette score.
    
    Args:
        embeddings: Array of embeddings
        min_clusters: Minimum number of clusters to try
        max_clusters: Maximum number of clusters to try
        step: Step size for cluster numbers
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (optimal_n_clusters, best_silhouette_score)
    """
    logger.info(f"Optimizing number of clusters from {min_clusters} to {max_clusters}")
    
    # Try different numbers of clusters
    cluster_range = range(min_clusters, max_clusters + 1, step)
    silhouette_scores = []
    
    for n_clusters in cluster_range:
        # Create and fit KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Calculate silhouette score
        score = silhouette_score(embeddings, cluster_labels)
        silhouette_scores.append(score)
        
        logger.info(f"Clusters: {n_clusters}, Silhouette: {score:.4f}")
    
    # Find optimal number of clusters
    best_idx = np.argmax(silhouette_scores)
    optimal_n_clusters = cluster_range[best_idx]
    best_score = silhouette_scores[best_idx]
    
    logger.info(f"Optimal number of clusters: {optimal_n_clusters} with silhouette score: {best_score:.4f}")
    
    # Visualize results
    plt.figure(figsize=(10, 6))
    plt.plot(list(cluster_range), silhouette_scores, 'o-')
    plt.axvline(x=optimal_n_clusters, color='r', linestyle='--')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs. Number of Clusters')
    plt.grid(True)
    plt.show()
    
    return optimal_n_clusters, best_score


def hybrid_clustering(company_df: pd.DataFrame, 
                     company_embeddings: np.ndarray,
                     taxonomy_embeddings: np.ndarray,
                     taxonomy_labels: List[str],
                     n_clusters: int = 50,
                     similarity_threshold: float = 0.5,
                     random_state: int = 42) -> pd.DataFrame:
    """
    Perform hybrid clustering and classification.
    
    This function clusters companies based on their embeddings,
    maps clusters to taxonomy labels, and assigns the labels to companies.
    
    Args:
        company_df: DataFrame with company data
        company_embeddings: Array of company embeddings
        taxonomy_embeddings: Array of taxonomy embeddings
        taxonomy_labels: List of taxonomy labels
        n_clusters: Number of clusters
        similarity_threshold: Threshold for assigning labels
        random_state: Random state for reproducibility
        
    Returns:
        DataFrame with assigned insurance labels
    """
    logger.info("Performing hybrid clustering and classification")
    
    # Create clustering model
    clustering = CompanyClustering(
        algorithm="kmeans",
        n_clusters=n_clusters,
        random_state=random_state
    )
    
    # Fit clustering model
    clustering.fit(company_embeddings)
    
    # Map clusters to taxonomy labels
    cluster_to_labels = clustering.map_clusters_to_taxonomy(
        taxonomy_embeddings, taxonomy_labels, top_k=5)
    
    # Assign labels to companies
    result_df = clustering.assign_cluster_labels(
        company_df, cluster_to_labels, similarity_threshold)
    
    # Count companies with labels
    has_label = sum(result_df['insurance_label'].str.len() > 0)
    total = len(result_df)
    coverage = has_label / total * 100
    
    logger.info(f"Classification results: {has_label}/{total} companies labeled ({coverage:.1f}%)")
    
    return result_df