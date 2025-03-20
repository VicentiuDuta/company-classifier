"""
Classifier module.

This module implements the insurance taxonomy classifier that assigns
insurance labels to companies based on text similarity.
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import pickle
import os
from tqdm import tqdm

# Configure logging
logger = logging.getLogger(__name__)

class InsuranceTaxonomyClassifier:
    """
    Classifier for matching companies to insurance taxonomy labels.
    """
    
    def __init__(self, 
                 similarity_threshold: float = 0.35,
                 top_k_labels: int = 5,
                 min_similarity_score: float = 0.25):
        """
        Initialize the classifier.
        
        Args:
            similarity_threshold: Threshold for considering a match
            top_k_labels: Maximum number of labels to assign
            min_similarity_score: Minimum similarity score to consider
        """
        self.similarity_threshold = similarity_threshold
        self.top_k_labels = top_k_labels
        self.min_similarity_score = min_similarity_score
        self.company_embeddings = None
        self.taxonomy_embeddings = None
        self.taxonomy_labels = None
        
    def fit(self, 
           company_embeddings: np.ndarray,
           taxonomy_embeddings: np.ndarray,
           taxonomy_labels: List[str],
           company_indices: Optional[List] = None):
        """
        Fit the classifier with embeddings.
        
        Args:
            company_embeddings: Array of company embeddings
            taxonomy_embeddings: Array of taxonomy embeddings
            taxonomy_labels: List of taxonomy labels
            company_indices: List of company indices or IDs
        """
        self.company_embeddings = company_embeddings
        self.taxonomy_embeddings = taxonomy_embeddings
        self.taxonomy_labels = taxonomy_labels
        self.company_indices = company_indices if company_indices else list(range(len(company_embeddings)))
        
        logger.info(f"Classifier fitted with {len(company_embeddings)} companies and {len(taxonomy_labels)} taxonomy labels")
        
    def predict(self, 
               query_embeddings: Optional[np.ndarray] = None,
               batch_size: int = 100) -> Dict:
        """
        Predict taxonomy labels for query embeddings or fitted company embeddings.
        
        Args:
            query_embeddings: Query embeddings to classify (if None, use fitted company embeddings)
            batch_size: Batch size for processing large datasets
            
        Returns:
            Dictionary with prediction results
        """
        if query_embeddings is None:
            if self.company_embeddings is None:
                raise ValueError("No company embeddings available. Call fit() first.")
            query_embeddings = self.company_embeddings
            query_indices = self.company_indices
        else:
            query_indices = list(range(len(query_embeddings)))
            
        logger.info(f"Predicting labels for {len(query_embeddings)} queries")
        
        results = {
            'query_indices': query_indices,
            'matched_labels': [],
            'similarity_scores': []
        }
        
        # Process in batches to avoid memory issues with large datasets
        for i in tqdm(range(0, len(query_embeddings), batch_size)):
            batch_embeddings = query_embeddings[i:i+batch_size]
            batch_results = self._predict_batch(batch_embeddings)
            
            results['matched_labels'].extend(batch_results['matched_labels'])
            results['similarity_scores'].extend(batch_results['similarity_scores'])
            
        return results
    
    def _predict_batch(self, query_embeddings: np.ndarray) -> Dict:
        """
        Predict taxonomy labels for a batch of query embeddings.
        
        Args:
            query_embeddings: Batch of query embeddings
            
        Returns:
            Dictionary with batch prediction results
        """
        # Normalize embeddings
        query_norm = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        taxonomy_norm = self.taxonomy_embeddings / np.linalg.norm(self.taxonomy_embeddings, axis=1, keepdims=True)
        
        # Calculate similarity matrix
        similarity_matrix = np.dot(query_norm, taxonomy_norm.T)
        
        batch_results = {
            'matched_labels': [],
            'similarity_scores': []
        }
        
        # For each query, find the matching labels
        for i in range(len(query_embeddings)):
            scores = similarity_matrix[i]
            
            # Find indices where similarity is above threshold
            matches = np.where(scores >= self.min_similarity_score)[0]
            
            # Sort by similarity score
            sorted_indices = matches[np.argsort(scores[matches])[::-1]]
            
            # Keep only top_k_labels
            top_indices = sorted_indices[:self.top_k_labels]
            
            # Get labels and scores
            query_labels = [self.taxonomy_labels[idx] for idx in top_indices]
            query_scores = scores[top_indices].tolist()
            
            # Apply final threshold
            final_labels = []
            final_scores = []
            
            for label, score in zip(query_labels, query_scores):
                if score >= self.similarity_threshold:
                    final_labels.append(label)
                    final_scores.append(score)
            
            batch_results['matched_labels'].append(final_labels)
            batch_results['similarity_scores'].append(final_scores)
            
        return batch_results
    
    def predict_for_texts(self, 
                        embedding_generator, 
                        texts: List[str],
                        batch_size: int = 32) -> Dict:
        """
        Predict taxonomy labels for text inputs.
        
        Args:
            embedding_generator: EmbeddingGenerator instance to create embeddings
            texts: List of texts to classify
            batch_size: Batch size for generating embeddings
            
        Returns:
            Dictionary with prediction results
        """
        # Generate embeddings for the texts
        embeddings = embedding_generator.generate_embeddings(texts, batch_size=batch_size)
        
        # Predict using the embeddings
        return self.predict(embeddings)
    
    def create_output_dataframe(self, 
                              company_df: pd.DataFrame,
                              prediction_results: Dict,
                              delimiter: str = '; ') -> pd.DataFrame:
        """
        Create output DataFrame with assigned insurance labels.
        
        Args:
            company_df: DataFrame with company information
            prediction_results: Results from the predict method
            delimiter: Delimiter for joining multiple labels
            
        Returns:
            DataFrame with added insurance_label column
        """
        # Create a copy of the company DataFrame
        result_df = company_df.copy()
        
        # Map prediction results to companies
        matched_labels_list = prediction_results['matched_labels']
        
        # Add insurance_label column
        result_df['insurance_label'] = [delimiter.join(labels) for labels in matched_labels_list]
        
        # Create formatted similarity scores for reference
        similarity_scores_list = prediction_results['similarity_scores']
        formatted_scores = []
        
        for labels, scores in zip(matched_labels_list, similarity_scores_list):
            if not labels:
                formatted_scores.append('')
            else:
                # Format each score with the corresponding label
                score_items = [f"{label} ({score:.2f})" for label, score in zip(labels, scores)]
                formatted_scores.append(delimiter.join(score_items))
                
        result_df['similarity_scores'] = formatted_scores
        
        return result_df
    
    def save(self, filepath: str):
        """
        Save the classifier to a file.
        
        Args:
            filepath: Path to save the classifier to
        """
        with open(filepath, 'wb') as f:
            pickle.dump({
                'similarity_threshold': self.similarity_threshold,
                'top_k_labels': self.top_k_labels,
                'min_similarity_score': self.min_similarity_score,
                'taxonomy_labels': self.taxonomy_labels,
                'taxonomy_embeddings': self.taxonomy_embeddings
            }, f)
        
        logger.info(f"Classifier saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """
        Load a classifier from a file.
        
        Args:
            filepath: Path to load the classifier from
            
        Returns:
            Loaded classifier instance
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        # Create a new instance
        classifier = cls(
            similarity_threshold=data['similarity_threshold'],
            top_k_labels=data['top_k_labels'],
            min_similarity_score=data['min_similarity_score']
        )
        
        # Set attributes
        classifier.taxonomy_labels = data['taxonomy_labels']
        classifier.taxonomy_embeddings = data['taxonomy_embeddings']
        
        logger.info(f"Classifier loaded from {filepath}")
        return classifier


def evaluate_classifier(classifier: InsuranceTaxonomyClassifier,
                       test_df: pd.DataFrame,
                       embedding_generator,
                       text_columns: List[str],
                       ground_truth_column: Optional[str] = None) -> Dict:
    """
    Evaluate the classifier on test data.
    
    Args:
        classifier: Trained classifier
        test_df: Test DataFrame
        embedding_generator: EmbeddingGenerator instance
        text_columns: Columns to use for classification
        ground_truth_column: Column with ground truth labels (if available)
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Combine text columns
    combined_texts = embedding_generator.combine_text_fields(test_df, text_columns)
    
    # Generate predictions
    prediction_results = classifier.predict_for_texts(embedding_generator, combined_texts)
    
    # Create output DataFrame
    output_df = classifier.create_output_dataframe(test_df, prediction_results)
    
    # Compute metrics
    metrics = {
        'num_samples': len(test_df),
        'samples_with_labels': sum(1 for labels in prediction_results['matched_labels'] if labels),
        'avg_labels_per_sample': np.mean([len(labels) for labels in prediction_results['matched_labels']]),
        'coverage': sum(1 for labels in prediction_results['matched_labels'] if labels) / len(test_df),
    }
    
    # Add metrics based on ground truth if available
    if ground_truth_column and ground_truth_column in test_df.columns:
        # TODO: Implement evaluation against ground truth
        pass
    
    return {
        'metrics': metrics,
        'output_df': output_df
    }


def optimize_threshold(classifier: InsuranceTaxonomyClassifier,
                      validation_df: pd.DataFrame,
                      embedding_generator,
                      text_columns: List[str],
                      thresholds: List[float] = [0.3, 0.4, 0.5, 0.6, 0.7],
                      target_labels_per_sample: float = 2.0) -> float:
    """
    Optimize the similarity threshold for the classifier.
    
    Args:
        classifier: Classifier to optimize
        validation_df: Validation DataFrame
        embedding_generator: EmbeddingGenerator instance
        text_columns: Columns to use for classification
        thresholds: List of thresholds to try
        target_labels_per_sample: Target average number of labels per sample
        
    Returns:
        Optimal threshold
    """
    # Combine text columns
    combined_texts = embedding_generator.combine_text_fields(validation_df, text_columns)
    
    # Generate embeddings once
    embeddings = embedding_generator.generate_embeddings(combined_texts)
    
    best_threshold = None
    best_diff = float('inf')
    
    logger.info("Optimizing similarity threshold")
    
    for threshold in thresholds:
        # Set current threshold
        classifier.similarity_threshold = threshold
        
        # Get predictions
        results = classifier.predict(embeddings)
        
        # Calculate average labels per sample
        avg_labels = np.mean([len(labels) for labels in results['matched_labels']])
        
        # Calculate difference from target
        diff = abs(avg_labels - target_labels_per_sample)
        
        logger.info(f"Threshold: {threshold}, Avg Labels: {avg_labels:.2f}, Diff: {diff:.2f}")
        
        if diff < best_diff:
            best_diff = diff
            best_threshold = threshold
    
    # Reset to best threshold
    classifier.similarity_threshold = best_threshold
    
    logger.info(f"Optimal threshold: {best_threshold}")
    
    return best_threshold