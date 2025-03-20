"""
XGBoost Classifier module.

This module implements an XGBoost-based classifier to enhance coverage 
of insurance taxonomy classification.
"""

import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
import os
from typing import List, Dict, Tuple

from src.features.embeddings import EmbeddingGenerator
from src.data.preprocessing import TextPreprocessor

# Configure logging
logger = logging.getLogger(__name__)

class XGBoostTaxonomyClassifier:
    """
    XGBoost-based classifier for insurance taxonomy classification.
    """
    
    def __init__(self, 
                 taxonomy_labels: List[str],
                 embedding_generator: EmbeddingGenerator,
                 max_tfidf_features: int = 500,
                 threshold_base: float = 0.5,
                 threshold_adjustment: float = 0.2):
        """
        Initialize the XGBoost classifier.
        
        Args:
            taxonomy_labels: List of taxonomy labels
            embedding_generator: EmbeddingGenerator instance
            max_tfidf_features: Maximum number of TF-IDF features
            threshold_base: Base threshold for prediction
            threshold_adjustment: Adjustment factor for threshold based on model accuracy
        """
        self.taxonomy_labels = taxonomy_labels
        self.embedding_generator = embedding_generator
        self.max_tfidf_features = max_tfidf_features
        self.threshold_base = threshold_base
        self.threshold_adjustment = threshold_adjustment
        
        self.tfidf = TfidfVectorizer(max_features=max_tfidf_features)
        self.mlb = MultiLabelBinarizer(classes=taxonomy_labels)
        self.models = []
        self.label_scores = []
        self.thresholds = []
        
    def prepare_features(self, df: pd.DataFrame, fit_tfidf: bool = False) -> np.ndarray:
        """
        Prepare features for training or prediction.
        
        Args:
            df: DataFrame with company data
            fit_tfidf: Whether to fit the TF-IDF vectorizer (for training) or just transform
            
        Returns:
            Array of combined features
        """
        # Combine text fields
        df['combined_text'] = df['description_processed'] + ' ' + df['business_tags_processed']
        
        # Add metadata if available
        if 'sector' in df.columns:
            df['combined_text'] += ' ' + df['sector'].fillna('')
        if 'category' in df.columns:
            df['combined_text'] += ' ' + df['category'].fillna('')
        if 'niche' in df.columns:
            df['combined_text'] += ' ' + df['niche'].fillna('')
        
        # Generate text features
        if fit_tfidf:
            X_tfidf = self.tfidf.fit_transform(df['combined_text'])
        else:
            X_tfidf = self.tfidf.transform(df['combined_text'])
        
        # Generate embeddings
        X_embeddings = self.embedding_generator.generate_embeddings(
            df['combined_text'].tolist(), show_progress=True)
        
        # Combine features
        X = np.hstack((X_tfidf.toarray(), X_embeddings))
        
        return X
    
    def fit(self, df_labeled: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
        """
        Train the XGBoost models.
        
        Args:
            df_labeled: DataFrame with labeled companies
            test_size: Proportion of data to use for validation
            random_state: Random state for reproducibility
        """
        logger.info("Preparing training data")
        
        # Transform labels to binary format
        labels = df_labeled['insurance_label'].str.split('; ').fillna('').tolist()
        y = self.mlb.fit_transform(labels)
        
        # Prepare features
        X = self.prepare_features(df_labeled, fit_tfidf=True)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state)
        
        # Train models
        logger.info(f"Training {len(self.taxonomy_labels)} XGBoost models")
        self.models = []
        self.label_scores = []
        
        for i, label in enumerate(self.taxonomy_labels):
            logger.info(f"Training model for label: {label}")
            
            dtrain = xgb.DMatrix(X_train, label=y_train[:, i])
            dval = xgb.DMatrix(X_val, label=y_val[:, i])
            
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'eta': 0.1,
                'max_depth': 5,
                'min_child_weight': 2,
                'gamma': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'seed': random_state
            }
            
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=100,
                evals=[(dval, 'validation')],
                early_stopping_rounds=10,
                verbose_eval=False
            )
            
            # Evaluate on validation set
            y_pred = model.predict(dval)
            y_pred_binary = (y_pred > 0.5).astype(int)
            accuracy = (y_pred_binary == y_val[:, i]).mean()
            
            logger.info(f"  Accuracy for {label}: {accuracy:.4f}")
            
            self.models.append(model)
            self.label_scores.append((label, accuracy))
            
            # Set adaptive threshold
            threshold = self.threshold_base + (1 - accuracy) * self.threshold_adjustment
            self.thresholds.append(threshold)
            
        logger.info("Training completed")
    
    def predict(self, df_unlabeled: pd.DataFrame) -> pd.DataFrame:
        """
        Predict labels for unlabeled companies.
        
        Args:
            df_unlabeled: DataFrame with unlabeled companies
            
        Returns:
            DataFrame with predictions
        """
        logger.info(f"Predicting labels for {len(df_unlabeled)} companies")
        
        # Prepare features
        X = self.prepare_features(df_unlabeled)
        
        # Make predictions
        predictions = np.zeros((len(df_unlabeled), len(self.taxonomy_labels)))
        
        for i, model in enumerate(self.models):
            dpredict = xgb.DMatrix(X)
            predictions[:, i] = model.predict(dpredict)
        
        # Apply thresholds
        predicted_labels = np.zeros_like(predictions, dtype=int)
        for i in range(predictions.shape[1]):
            predicted_labels[:, i] = (predictions[:, i] >= self.thresholds[i]).astype(int)
        
        # Transform predictions back to text labels
        predicted_label_lists = self.mlb.inverse_transform(predicted_labels)
        
        # Create output dataframe
        df_result = df_unlabeled.copy()
        
        # Add predictions to dataframe
        df_result['insurance_label'] = [
            '; '.join(labels) if labels else None 
            for labels in predicted_label_lists
        ]
        
        # Add similarity scores for reference
        df_result['similarity_scores'] = [
            '; '.join([f"{label} ({score:.2f})" 
                     for label, score in zip(labels, predictions[j][np.where(predicted_labels[j])]) 
                     if len(labels) > 0]) 
            if len(labels) > 0 else None
            for j, labels in enumerate(predicted_label_lists)
        ]
        
        logger.info(f"Prediction completed, {sum(df_result['insurance_label'].notna())} companies labeled")
        
        return df_result
