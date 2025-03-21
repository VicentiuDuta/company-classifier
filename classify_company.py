#!/usr/bin/env python3
"""
Interactive classification script for a single company.

This script allows users to input a company description and business tags
and get insurance taxonomy classification in real-time.

Usage:
    python classify_company.py
"""

import argparse
import pickle
import os
import logging
import yaml

from src.data.preprocessing import TextPreprocessor
from src.features.embeddings import EmbeddingGenerator
from src.models.classifier import InsuranceTaxonomyClassifier

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("company_classifier")

def load_model(model_dir="models", config_path="config/config.yaml"):
    """Load the trained model components."""
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load classifier
    classifier_path = os.path.join(model_dir, "classifier.pkl")
    if not os.path.exists(classifier_path):
        raise FileNotFoundError(f"Classifier model not found at {classifier_path}")
    
    classifier = InsuranceTaxonomyClassifier.load(classifier_path)
    
    # Create embedding generator
    embed_config = config.get("embeddings", {})
    embedding_generator = EmbeddingGenerator(
        model_name=embed_config.get("model_name", "all-MiniLM-L6-v2"),
        max_seq_length=embed_config.get("max_seq_length", 128)
    )
    
    # Create text preprocessor
    preproc_config = config.get("preprocessing", {})
    preprocessor = TextPreprocessor(
        remove_stopwords=preproc_config.get("remove_stopwords", True),
        min_word_length=preproc_config.get("min_word_length", 2),
        lemmatize=preproc_config.get("lemmatize", True)
    )
    
    return {
        "classifier": classifier,
        "embedding_generator": embedding_generator,
        "preprocessor": preprocessor
    }

def classify_company(description, business_tags, sector=None, category=None, niche=None, model_components=None):
    """Classify a single company based on input text."""
    if model_components is None:
        model_components = load_model()
    
    classifier = model_components["classifier"]
    embedding_generator = model_components["embedding_generator"]
    preprocessor = model_components["preprocessor"]
    
    # Preprocess input
    processed_description = preprocessor.preprocess_text(description)
    processed_tags = preprocessor.preprocess_text(business_tags)
    
    # Create a mini dataframe with the company info
    import pandas as pd
    company_data = {
        'description': [description],
        'business_tags': [business_tags],
        'description_processed': [processed_description],
        'business_tags_processed': [processed_tags],
        'sector': [sector] if sector else [None],
        'category': [category] if category else [None],
        'niche': [niche] if niche else [None]
    }
    company_df = pd.DataFrame(company_data)
    
    # Generate text representation
    text = embedding_generator.combine_text_fields(
        company_df, ['description_processed', 'business_tags_processed'], 
        weights=[1.0, 0.5]
    )[0]
    
    # Generate embeddings
    embeddings = embedding_generator.generate_embeddings([text], show_progress=False)
    
    # Classify
    prediction_results = classifier.predict(embeddings)
    
    # Format results
    labels = prediction_results['matched_labels'][0]
    scores = prediction_results['similarity_scores'][0]
    
    result = []
    for label, score in zip(labels, scores):
        result.append((label, score))
    
    return result

def main():
    """Main function for interactive company classification."""
    print("Loading classification model...")
    model_components = load_model()
    print("Model loaded successfully!")
    
    while True:
        print("\n" + "="*50)
        print("COMPANY CLASSIFICATION")
        print("="*50)
        
        description = input("Enter company description (or 'quit' to exit): ")
        if description.lower() == 'quit':
            break
            
        business_tags = input("Enter business tags (comma separated): ")
        
        # Optional fields
        print("\nOptional information (press Enter to skip):")
        sector = input("Sector: ")
        category = input("Category: ")
        niche = input("Niche: ")
        
        if not description:
            print("Description is required. Please try again.")
            continue
        
        # Classify
        print("\nClassifying company...")
        results = classify_company(
            description, business_tags, sector, category, niche, model_components
        )
        
        if results:
            print("\nClassification Results:")
            print("-" * 40)
            for label, score in results:
                print(f"{label:<30} (confidence: {score:.2f})")
            print("-" * 40)
        else:
            print("\nUnable to classify this company with sufficient confidence.")
            print("Try providing more detailed description or business tags.")
        
        print("\nPress Enter to classify another company or type 'quit' to exit.")
        if input().lower() == 'quit':
            break
    
    print("Thank you for using the Company Classifier!")

if __name__ == "__main__":
    main()
