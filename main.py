#!/usr/bin/env python3
"""
Main entry point for the insurance taxonomy classifier.
"""
import argparse
import logging
import yaml
import os
import pandas as pd
from typing import Dict, List, Optional
import numpy as np
import time

from src.data.preprocessing import TextPreprocessor, load_and_preprocess_data
from src.features.embeddings import EmbeddingGenerator
from src.models.classifier import InsuranceTaxonomyClassifier, optimize_threshold
from src.utils.evaluation import generate_evaluation_report
from src.utils.visualization import generate_report_dashboard


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Set up logging configuration."""
    log_level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    
    level = log_level_map.get(log_level.upper(), logging.INFO)
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    return logging.getLogger("insurance_classifier")


def train_model(config: Dict, logger: logging.Logger) -> Dict:
    """
    Train the insurance taxonomy classifier.
    
    Args:
        config: Configuration dictionary
        logger: Logger instance
        
    Returns:
        Dictionary with trained model components
    """
    # Extract paths from config
    company_file = config.get("data", {}).get("company_file", "data/raw/ml_insurance_challenge.csv")
    taxonomy_file = config.get("data", {}).get("taxonomy_file", "data/raw/insurance_taxonomy.csv")
    
    # Create preprocessor
    preproc_config = config.get("preprocessing", {})
    preprocessor = TextPreprocessor(
        remove_stopwords=preproc_config.get("remove_stopwords", True),
        min_word_length=preproc_config.get("min_word_length", 2),
        lemmatize=preproc_config.get("lemmatize", True)
    )
    
    # Load and preprocess data
    logger.info("Loading and preprocessing data")
    data = load_and_preprocess_data(company_file, taxonomy_file, preprocessor)
    
    company_df = data["company_df"]
    taxonomy_labels = data["taxonomy_labels"]
    preprocessed_taxonomy = data["preprocessed_taxonomy"]
    
    # Create embedding generator
    embed_config = config.get("embeddings", {})
    embedding_generator = EmbeddingGenerator(
        model_name=embed_config.get("model_name", "all-MiniLM-L6-v2"),
        max_seq_length=embed_config.get("max_seq_length", 128),
        cache_dir=embed_config.get("cache_dir", "data/embeddings")
    )
    
    # Generate company embeddings
    logger.info("Generating company embeddings")
    text_columns = config.get("data", {}).get("text_columns", ["description", "business_tags"])
    company_texts = embedding_generator.combine_text_fields(
        company_df, text_columns, weights=[1.0, 0.5])
    
    company_embeddings_file = os.path.join(
        embed_config.get("cache_dir", "data/embeddings"), 
        "company_embeddings.pkl"
    )
    
    company_embeddings = embedding_generator.generate_embeddings(company_texts)
    
    # Generate taxonomy embeddings
    logger.info("Generating taxonomy embeddings")
    taxonomy_embeddings_file = os.path.join(
        embed_config.get("cache_dir", "data/embeddings"), 
        "taxonomy_embeddings.pkl"
    )
    
    taxonomy_embeddings = embedding_generator.generate_embeddings(taxonomy_labels)
    
    # Create and fit classifier
    logger.info("Fitting classifier")
    classifier_config = config.get("classification", {})
    classifier = InsuranceTaxonomyClassifier(
        similarity_threshold=classifier_config.get("similarity_threshold", 0.5),
        top_k_labels=classifier_config.get("top_k_labels", 3),
        min_similarity_score=classifier_config.get("min_similarity_threshold", 0.3)
    )
    
    classifier.fit(company_embeddings, taxonomy_embeddings, taxonomy_labels)
    
    # Optimize similarity threshold if enabled
    if config.get("optimization", {}).get("optimize_threshold", True):
        logger.info("Optimizing similarity threshold")
        validate_size = config.get("optimization", {}).get("validation_size", 500)
        
        # Use a subset for threshold optimization
        validate_df = company_df.sample(min(validate_size, len(company_df)))
        validate_texts = embedding_generator.combine_text_fields(
            validate_df, text_columns, weights=[1.0, 0.5])
        validate_embeddings = embedding_generator.generate_embeddings(validate_texts)
        
        # Get threshold values to try
        thresholds = config.get("optimization", {}).get(
            "thresholds", [0.3, 0.4, 0.5, 0.6, 0.7])
        
        # Optimize threshold
        optimal_threshold = optimize_threshold(
            classifier, validate_df, embedding_generator, 
            text_columns, thresholds)
    
    # Save model components
    model_dir = config.get("paths", {}).get("model_dir", "models")
    os.makedirs(model_dir, exist_ok=True)
    
    classifier.save(os.path.join(model_dir, "classifier.pkl"))
    
    logger.info("Model training completed")
    
    return {
        "classifier": classifier,
        "embedding_generator": embedding_generator,
        "company_df": company_df,
        "company_embeddings": company_embeddings,
        "taxonomy_embeddings": taxonomy_embeddings,
        "taxonomy_labels": taxonomy_labels
    }


def evaluate_model(model_components: Dict, config: Dict, logger: logging.Logger) -> Dict:
    """
    Evaluate the trained model.
    
    Args:
        model_components: Dictionary with model components
        config: Configuration dictionary
        logger: Logger instance
        
    Returns:
        Dictionary with evaluation results
    """
    logger.info("Evaluating model")
    
    # Extract components
    classifier = model_components["classifier"]
    embedding_generator = model_components["embedding_generator"]
    company_df = model_components["company_df"]
    company_embeddings = model_components["company_embeddings"]
    taxonomy_labels = model_components["taxonomy_labels"]
    
    # Run predictions
    logger.info("Generating predictions")
    start_time = time.time()
    prediction_results = classifier.predict(company_embeddings)
    prediction_time = time.time() - start_time
    
    logger.info(f"Prediction completed in {prediction_time:.2f} seconds "
               f"for {len(company_embeddings)} companies")
    
    # Create output DataFrame
    output_df = classifier.create_output_dataframe(company_df, prediction_results)
    
    # Generate evaluation report
    logger.info("Generating evaluation report")
    results_dir = config.get("paths", {}).get("results_dir", "results")
    generate_evaluation_report(output_df, prediction_results, results_dir)
    
    # Generate dashboard
    logger.info("Generating visualization dashboard")
    dashboard_dir = os.path.join(results_dir, "dashboard")
    generate_report_dashboard(
        output_df, 
        prediction_results, 
        dashboard_dir,
        taxonomy_labels
    )
    
    # Save results
    logger.info("Saving results")
    output_file = config.get("paths", {}).get(
        "output_file", "results/classified_companies.csv")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    output_df.to_csv(output_file, index=False)
    
    logger.info(f"Results saved to {output_file}")
    
    # Calculate metrics
    metrics = {
        "num_companies": len(company_df),
        "num_taxonomy_labels": len(taxonomy_labels),
        "companies_with_labels": sum(1 for labels in prediction_results["matched_labels"] if labels),
        "avg_labels_per_company": np.mean([len(labels) for labels in prediction_results["matched_labels"]]),
        "coverage_percentage": sum(1 for labels in prediction_results["matched_labels"] if labels) / len(company_df) * 100,
        "prediction_time_seconds": prediction_time,
        "predictions_per_second": len(company_embeddings) / prediction_time
    }
    
    logger.info(f"Evaluation metrics: {metrics}")
    
    return {
        "metrics": metrics,
        "output_df": output_df,
        "prediction_results": prediction_results
    }


def predict_for_new_data(input_file: str, output_file: str, 
                       model_dir: str, config: Dict, 
                       logger: logging.Logger) -> None:
    """
    Run predictions for new data.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        model_dir: Directory with saved model components
        config: Configuration dictionary
        logger: Logger instance
    """
    logger.info(f"Running predictions for {input_file}")
    
    # Load classifier
    classifier_path = os.path.join(model_dir, "classifier.pkl")
    logger.info(f"Loading classifier from {classifier_path}")
    classifier = InsuranceTaxonomyClassifier.load(classifier_path)
    
    # Load input data
    logger.info(f"Loading input data from {input_file}")
    input_df = pd.read_csv(input_file)
    
    # Create preprocessor
    preproc_config = config.get("preprocessing", {})
    preprocessor = TextPreprocessor(
        remove_stopwords=preproc_config.get("remove_stopwords", True),
        min_word_length=preproc_config.get("min_word_length", 2),
        lemmatize=preproc_config.get("lemmatize", True)
    )
    
    # Preprocess input data
    logger.info("Preprocessing input data")
    text_columns = config.get("data", {}).get("text_columns", ["description", "business_tags"])
    input_df = preprocessor.preprocess_dataframe(input_df, text_columns)
    
    # Create embedding generator
    embed_config = config.get("embeddings", {})
    embedding_generator = EmbeddingGenerator(
        model_name=embed_config.get("model_name", "all-MiniLM-L6-v2"),
        max_seq_length=embed_config.get("max_seq_length", 128)
    )
    
    # Generate predictions
    logger.info("Generating predictions")
    prediction_results = classifier.predict_for_texts(
        embedding_generator,
        embedding_generator.combine_text_fields(
            input_df, text_columns, weights=[1.0, 0.5])
    )
    
    # Create output DataFrame
    output_df = classifier.create_output_dataframe(input_df, prediction_results)
    
    # Save results
    logger.info(f"Saving results to {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    output_df.to_csv(output_file, index=False)
    
    logger.info("Prediction completed")


def main():
    """Main function of the application."""
    parser = argparse.ArgumentParser(description="Insurance Taxonomy Classifier")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--mode", type=str, choices=["train", "evaluate", "predict"], 
                        default="train", help="Operating mode")
    parser.add_argument("--input_file", type=str, help="Input file for predictions")
    parser.add_argument("--output_file", type=str, help="Output file for results")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging(args.log_level)
    
    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Running in mode: {args.mode}")
    
    if args.mode == "train":
        # Train model
        model_components = train_model(config, logger)
        
        # Evaluate if requested
        if config.get("evaluation", {}).get("evaluate_after_training", True):
            evaluate_model(model_components, config, logger)
            
    elif args.mode == "evaluate":
        # Load model components
        model_dir = config.get("paths", {}).get("model_dir", "models")
        classifier = InsuranceTaxonomyClassifier.load(os.path.join(model_dir, "classifier.pkl"))
        
        # TODO: Load other components and run evaluation
        logger.warning("Evaluate mode not fully implemented yet")
        
    elif args.mode == "predict":
        # Check required arguments
        if not args.input_file or not args.output_file:
            logger.error("For predict mode, --input_file and --output_file are required")
            return
        
        # Run predictions
        model_dir = config.get("paths", {}).get("model_dir", "models")
        predict_for_new_data(args.input_file, args.output_file, model_dir, config, logger)
    
    logger.info("Process completed successfully")


if __name__ == "__main__":
    main()