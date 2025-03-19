#!/usr/bin/env python3
"""
Test script for the insurance taxonomy classifier.
"""
import logging
import pandas as pd
import numpy as np
import os
import sys
from tqdm import tqdm

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.preprocessing import TextPreprocessor, load_data, prepare_company_features, prepare_taxonomy_features
from src.features.embeddings import EmbeddingGenerator
from src.models.classifier import InsuranceTaxonomyClassifier

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Test the classifier on a small dataset."""
    try:
        # Load the data
        logger.info("Loading data...")
        companies_df, taxonomy_df = load_data(
            "data/raw/ml_insurance_challenge.csv",
            "data/raw/insurance_taxonomy.csv"
        )
        
        # Print data overview
        logger.info(f"Loaded {len(companies_df)} companies and {len(taxonomy_df)} taxonomy labels")
        print("\nSample companies data:")
        print(companies_df.head(2))
        print("\nSample taxonomy data:")
        print(taxonomy_df.head(2))
        
        # Initialize preprocessor
        logger.info("Initializing text preprocessor...")
        config = {
            'remove_stopwords': True,
            'min_word_length': 2,
            'lemmatize': True
        }
        preprocessor = TextPreprocessor(config)
        
        # Preprocess data
        logger.info("Preprocessing company data...")
        processed_companies = prepare_company_features(companies_df, preprocessor)
        
        logger.info("Preprocessing taxonomy data...")
        processed_taxonomy = prepare_taxonomy_features(taxonomy_df, preprocessor)
        
        # Sample test with a subset for faster testing
        logger.info("Creating a small test subset...")
        test_companies = processed_companies.head(100).copy()
        test_taxonomy = processed_taxonomy.copy()
        
        # Initialize embedding generator
        logger.info("Initializing embedding generator...")
        embedding_config = {
            'model_name': 'all-MiniLM-L6-v2',
            'max_seq_length': 128
        }
        embedding_gen = EmbeddingGenerator(embedding_config)
        
        # Initialize classifier
        logger.info("Initializing classifier...")
        classifier_config = {
            'similarity_threshold': 0.45,
            'top_k_labels': 3
        }
        classifier = InsuranceTaxonomyClassifier(embedding_gen, classifier_config)
        
        # Prepare taxonomy and companies
        logger.info("Preparing taxonomy embeddings...")
        classifier.prepare_taxonomy(test_taxonomy)
        
        logger.info("Preparing company embeddings...")
        classifier.prepare_companies(test_companies)
        
        # Classify companies
        logger.info("Classifying companies...")
        results_df = classifier.classify_companies()
        
        # Show results
        print("\nClassification results:")
        for i, row in results_df.head(10).iterrows():
            print(f"\nCompany: {row['description'][:100]}...")
            print(f"Sector: {row['sector']}")
            print(f"Category: {row['category']}")
            print(f"Predicted insurance labels: {row['insurance_label']}")
            print(f"Label scores: {row['label_scores']}")
        
        # Test a single prediction
        test_text = "A company providing health insurance for individuals and families"
        logger.info(f"Testing single prediction for: '{test_text}'")
        
        # Preprocess the test text
        processed_text = preprocessor.clean_text(test_text)
        
        # Predict
        predictions = classifier.predict_for_text(processed_text)
        
        print("\nSingle text prediction:")
        print(f"Text: {test_text}")
        print(f"Processed: {processed_text}")
        print("Predictions:")
        for label, score in predictions:
            print(f"  - {label}: {score:.4f}")
        
        logger.info("Test completed successfully")
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()