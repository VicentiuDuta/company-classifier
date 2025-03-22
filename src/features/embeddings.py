"""
Embeddings module.

This module implements functionality for creating and managing embeddings
for text data using sentence-transformers.
"""

import os
import logging
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch
import pickle
from typing import List, Dict, Union, Optional

# Configure logging
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """
    Class for generating and managing text embeddings using sentence-transformers.
    """
    
    def __init__(self, model_name: str = "all-mpnet-base-v2", 
                 max_seq_length: int = 256,
                 cache_dir: str = None,
                 device: str = None):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the sentence-transformers model to use
            max_seq_length: Maximum sequence length for the model
            cache_dir: Directory to cache the embeddings
            device: Device to use for computation (cpu or cuda)
        """
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.cache_dir = cache_dir
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Initializing embedding model {model_name} on {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.model.max_seq_length = max_seq_length
        
        # Create cache directory if it doesn't exist
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
    def generate_embeddings(self, texts: List[str], 
                           batch_size: int = 32, 
                           show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to generate embeddings for
            batch_size: Batch size for embedding generation
            show_progress: Whether to show a progress bar
            
        Returns:
            Array of embeddings
        """
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        # Use tqdm if show_progress is True
        if show_progress:
            embeddings = self.model.encode(texts, batch_size=batch_size, 
                                         show_progress_bar=True)
        else:
            embeddings = self.model.encode(texts, batch_size=batch_size, 
                                         show_progress_bar=False)
            
        return embeddings
    
    def combine_text_fields(self, df: pd.DataFrame, 
                      text_columns: List[str], 
                      weights: Optional[List[float]] = None) -> List[str]:
        """
        An advanced version for combining texts that adds explicit context.
        """
        if weights is None:
            weights = [1.0] * len(text_columns)
            
        combined_texts = []
        
        for _, row in df.iterrows():
            # Add structured context
            context_parts = []
            
            # Include structured metadata about the company
            if 'sector' in df.columns and pd.notna(row['sector']):
                context_parts.append(f"Sector: {row['sector']}")
            if 'category' in df.columns and pd.notna(row['category']):
                context_parts.append(f"Category: {row['category']}")
            if 'niche' in df.columns and pd.notna(row['niche']):
                context_parts.append(f"Niche: {row['niche']}")
                
            # Add the actual texts with various weights
            text_parts = []
            for col, weight in zip(text_columns, weights):
                if pd.notna(row[col]) and row[col]:
                    text = str(row[col])
                    # Repeat the text multiple times to emphasize importance
                    text = " ".join([text] * int(weight))
                    text_parts.append(text)
            
            # Combine everything, putting context first to frame the text
            full_text = " ".join(context_parts + text_parts)
            combined_texts.append(full_text)
            
        return combined_texts
    
    def save_model(self, path: str):
        """
        Save the sentence transformer model to disk.
        
        Args:
            path: Path to save the model to
        """
        self.model.save(path)
        
    def load_model(self, path: str):
        """
        Load a sentence transformer model from disk.
        
        Args:
            path: Path to load the model from
        """
        self.model = SentenceTransformer(path)


def calculate_similarity(embedding1: np.ndarray, 
                        embedding2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two embeddings.
    
    Args:
        embedding1: First embedding
        embedding2: Second embedding
        
    Returns:
        Cosine similarity (between -1 and 1)
    """
    # Normalize embeddings
    embedding1_norm = embedding1 / np.linalg.norm(embedding1)
    embedding2_norm = embedding2 / np.linalg.norm(embedding2)
    
    # Calculate cosine similarity
    similarity = np.dot(embedding1_norm, embedding2_norm)
    
    return similarity


def calculate_similarity_matrix(embeddings1: np.ndarray, 
                              embeddings2: np.ndarray) -> np.ndarray:
    """
    Calculate pairwise cosine similarity between two sets of embeddings.
    
    Args:
        embeddings1: First set of embeddings (n x d)
        embeddings2: Second set of embeddings (m x d)
        
    Returns:
        Similarity matrix (n x m)
    """
    # Normalize embeddings
    embeddings1_norm = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
    embeddings2_norm = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
    
    # Calculate similarity matrix
    similarity_matrix = np.dot(embeddings1_norm, embeddings2_norm.T)
    
    return similarity_matrix


def find_top_k_similar(query_embedding: np.ndarray, 
                     corpus_embeddings: np.ndarray, 
                     k: int = 5) -> List[int]:
    """
    Find the top k most similar embeddings to a query embedding.
    
    Args:
        query_embedding: Query embedding
        corpus_embeddings: Corpus of embeddings to search in
        k: Number of top matches to return
        
    Returns:
        List of indices of the top k most similar embeddings
    """
    # Calculate similarity scores
    similarities = [calculate_similarity(query_embedding, emb) for emb in corpus_embeddings]
    
    # Get indices of top k similarities
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    
    return top_k_indices.tolist()