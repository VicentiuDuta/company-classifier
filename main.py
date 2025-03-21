#!/usr/bin/env python3
"""
Unified Classification Script for Insurance Taxonomy.

This script combines the functionality of main.py, reclassify_unlabeled.py, 
and sector_based_classification.py into a single workflow.

Usage:
    python unified_classification.py --config config/config.yaml
"""

import argparse
import logging
import yaml
import os
import pandas as pd
import numpy as np
import time
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Set, Optional
from tqdm import tqdm

# Import modules from project
from src.data.preprocessing import TextPreprocessor, load_and_preprocess_data
from src.features.embeddings import EmbeddingGenerator
from src.models.classifier import InsuranceTaxonomyClassifier, optimize_threshold
from src.utils.evaluation import generate_evaluation_report
from src.utils.visualization import generate_report_dashboard

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("insurance_classifier")

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

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise

def train_initial_model(config: Dict, logger: logging.Logger) -> Dict:
    """
    Train the insurance taxonomy classifier using similarity-based approach.
    
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
    
    company_embeddings = embedding_generator.generate_embeddings(company_texts)
    
    # Generate taxonomy embeddings
    logger.info("Generating taxonomy embeddings")
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

def initial_classification(model_components: Dict, config: Dict, logger: logging.Logger) -> Dict:
    """
    Perform initial classification on the dataset.
    
    Args:
        model_components: Dictionary with model components
        config: Configuration dictionary
        logger: Logger instance
        
    Returns:
        Dictionary with classification results and output DataFrame
    """
    # Extract components
    classifier = model_components["classifier"]
    company_df = model_components["company_df"]
    company_embeddings = model_components["company_embeddings"]
    
    # Run predictions
    logger.info("Generating initial predictions")
    start_time = time.time()
    prediction_results = classifier.predict(company_embeddings)
    prediction_time = time.time() - start_time
    
    logger.info(f"Prediction completed in {prediction_time:.2f} seconds "
               f"for {len(company_embeddings)} companies")
    
    # Create output DataFrame
    output_df = classifier.create_output_dataframe(company_df, prediction_results)
    
    # Save initial results
    results_dir = config.get("paths", {}).get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, "classified_companies_initial.csv")
    output_df.to_csv(output_file, index=False)
    
    logger.info(f"Initial classification results saved to {output_file}")
    
    # Calculate coverage metrics
    labeled_mask = output_df['insurance_label'].notna() & (output_df['insurance_label'] != '')
    labeled_count = labeled_mask.sum()
    coverage_percentage = labeled_count / len(output_df) * 100
    
    logger.info(f"Initial coverage: {labeled_count}/{len(output_df)} companies "
               f"({coverage_percentage:.2f}%)")
    
    return {
        "output_df": output_df,
        "prediction_results": prediction_results,
        "coverage_percentage": coverage_percentage
    }

def reclassify_unlabeled(output_df: pd.DataFrame, 
                       classifier: InsuranceTaxonomyClassifier, 
                       embedding_generator: EmbeddingGenerator, 
                       text_columns: List[str],
                       logger: logging.Logger) -> pd.DataFrame:
    """
    Reclassify unlabeled companies with lower thresholds.
    
    Args:
        output_df: DataFrame with initial classification results
        classifier: Trained classifier
        embedding_generator: Embedding generator instance
        text_columns: Text columns to use
        logger: Logger instance
        
    Returns:
        DataFrame with updated classification results
    """
    logger.info("Starting reclassification of unlabeled companies")
    
    # Identify unlabeled companies
    unlabeled_mask = output_df['insurance_label'].isna() | (output_df['insurance_label'] == '')
    unlabeled_companies = output_df[unlabeled_mask]
    
    logger.info(f"Found {len(unlabeled_companies)} unlabeled companies "
               f"({len(unlabeled_companies)/len(output_df)*100:.2f}%)")
    
    if len(unlabeled_companies) == 0:
        logger.info("No unlabeled companies to reclassify")
        return output_df
    
    # Store original thresholds
    original_threshold = classifier.similarity_threshold
    original_min_score = classifier.min_similarity_score
    original_top_k = classifier.top_k_labels
    
    # Set lower thresholds for better coverage
    classifier.similarity_threshold = 0.30
    classifier.min_similarity_score = 0.20
    classifier.top_k_labels = 5
    
    logger.info(f"Adjusted thresholds - Similarity: {classifier.similarity_threshold}, "
               f"Min score: {classifier.min_similarity_score}, Top K: {classifier.top_k_labels}")
    
    # Prepare texts for unlabeled companies
    unlabeled_texts = embedding_generator.combine_text_fields(
        unlabeled_companies, text_columns, weights=[1.0, 0.5])
    
    # Generate predictions
    prediction_results = classifier.predict_for_texts(
        embedding_generator, unlabeled_texts)
    
    # Create output DataFrame for unlabeled companies
    unlabeled_output = classifier.create_output_dataframe(
        unlabeled_companies, prediction_results)
    
    # Check how many unlabeled companies got labels now
    newly_labeled = unlabeled_output['insurance_label'].notna() & (unlabeled_output['insurance_label'] != '')
    newly_labeled_count = newly_labeled.sum()
    
    logger.info(f"Newly labeled companies: {newly_labeled_count}/{len(unlabeled_companies)} "
               f"({newly_labeled_count/len(unlabeled_companies)*100:.2f}%)")
    
    # Update the original DataFrame
    result_df = output_df.copy()
    unlabeled_indices = unlabeled_companies.index
    
    result_df.loc[unlabeled_indices, 'insurance_label'] = unlabeled_output['insurance_label']
    result_df.loc[unlabeled_indices, 'similarity_scores'] = unlabeled_output['similarity_scores']
    
    # Restore original thresholds
    classifier.similarity_threshold = original_threshold
    classifier.min_similarity_score = original_min_score
    classifier.top_k_labels = original_top_k
    
    # Calculate new coverage
    labeled_mask = result_df['insurance_label'].notna() & (result_df['insurance_label'] != '')
    labeled_count = labeled_mask.sum()
    coverage_percentage = labeled_count / len(result_df) * 100
    
    logger.info(f"Coverage after reclassification: {labeled_count}/{len(result_df)} companies "
               f"({coverage_percentage:.2f}%)")
    
    return result_df

def build_sector_mappings(df: pd.DataFrame, 
                      label_column: str = 'insurance_label',
                      delimiter: str = '; ', 
                      min_count: int = 2) -> Dict:
    """
    Build mappings between sectors/categories/niches and insurance labels.
    
    Args:
        df: DataFrame with classified companies
        label_column: Name of the label column
        delimiter: Delimiter for multiple labels
        min_count: Minimum occurrences to include a label
        
    Returns:
        Dictionary with mappings
    """
    # Use only companies that already have labels
    labeled_df = df[df[label_column].notna() & (df[label_column] != '')]
    
    # Initialize dictionaries
    sector_to_labels = defaultdict(Counter)
    category_to_labels = defaultdict(Counter)
    niche_to_labels = defaultdict(Counter)
    
    # Build mappings
    for _, row in labeled_df.iterrows():
        if pd.isna(row[label_column]) or row[label_column] == '':
            continue
            
        labels = row[label_column].split(delimiter)
        
        # Sector -> labels mapping
        if pd.notna(row.get('sector', None)):
            for label in labels:
                sector_to_labels[row['sector']][label] += 1
        
        # Category -> labels mapping
        if pd.notna(row.get('category', None)):
            for label in labels:
                category_to_labels[row['category']][label] += 1
        
        # Niche -> labels mapping
        if pd.notna(row.get('niche', None)):
            for label in labels:
                niche_to_labels[row['niche']][label] += 1
    
    # Filter rare labels
    for sector in sector_to_labels:
        sector_to_labels[sector] = {k: v for k, v in sector_to_labels[sector].items() if v >= min_count}
    
    for category in category_to_labels:
        category_to_labels[category] = {k: v for k, v in category_to_labels[category].items() if v >= min_count}
    
    for niche in niche_to_labels:
        niche_to_labels[niche] = {k: v for k, v in niche_to_labels[niche].items() if v >= min_count}
    
    return {
        'sector_to_labels': dict(sector_to_labels),
        'category_to_labels': dict(category_to_labels),
        'niche_to_labels': dict(niche_to_labels)
    }

def predict_by_sector(company, mappings: Dict, top_k: int = 3) -> Tuple[List[str], List[float]]:
    """
    Predict labels for a company based on sector, category, and niche.
    
    Args:
        company: Row from DataFrame representing a company
        mappings: Dictionary with built mappings
        top_k: Maximum number of labels to return
        
    Returns:
        Tuple of (predicted_labels, scores)
    """
    candidate_labels = Counter()
    
    # Add labels based on sector
    sector = company.get('sector')
    if pd.notna(sector) and sector in mappings['sector_to_labels']:
        for label, count in mappings['sector_to_labels'][sector].items():
            candidate_labels[label] += count * 1  # Standard weight for sector
    
    # Add labels based on category
    category = company.get('category')
    if pd.notna(category) and category in mappings['category_to_labels']:
        for label, count in mappings['category_to_labels'][category].items():
            candidate_labels[label] += count * 2  # Double weight for category (more specific)
    
    # Add labels based on niche
    niche = company.get('niche')
    if pd.notna(niche) and niche in mappings['niche_to_labels']:
        for label, count in mappings['niche_to_labels'][niche].items():
            candidate_labels[label] += count * 3  # Triple weight for niche (most specific)
    
    # Return top k labels
    top_labels = []
    top_scores = []
    
    max_score = max(candidate_labels.values()) if candidate_labels else 1
    
    for label, score in candidate_labels.most_common(top_k):
        top_labels.append(label)
        normalized_score = 0.30 + (0.35 * score / max_score)
        top_scores.append(normalized_score)
    
    return top_labels, top_scores

def sector_based_classification(df: pd.DataFrame, 
                          min_confidence: float = 0.3, 
                          top_k: int = 3, 
                          delimiter: str = '; ',
                          logger: logging.Logger = None) -> pd.DataFrame:
    """
    Apply sector-based classification for unlabeled or poorly classified companies.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    logger.info("Starting sector-based classification")
    
    # Build mappings
    logger.info("Building sector->label mappings")
    mappings = build_sector_mappings(df, min_count=2)
    
    # Log mapping statistics
    logger.info(f"Sectors mapped: {len(mappings['sector_to_labels'])}")
    logger.info(f"Categories mapped: {len(mappings['category_to_labels'])}")
    logger.info(f"Niches mapped: {len(mappings['niche_to_labels'])}")
    
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Add column for classification method
    if 'classification_method' not in result_df.columns:
        result_df['classification_method'] = 'semantic'
    
    # Counters for statistics
    stats = {'total': 0, 'improved': 0, 'new': 0}
    
    # Apply classification for each company
    logger.info("Applying sector-based classification")
    for idx, row in tqdm(result_df.iterrows(), total=len(result_df), desc="Processing companies"):
        if pd.isna(row['insurance_label']) or row['insurance_label'] == '':
            # Unlabeled company
            labels, scores = predict_by_sector(row, mappings, top_k)
            
            if labels and scores[0] >= min_confidence:
                # Convert scores to formatted strings
                score_strings = [f"{score:.2f}" for score in scores]
                
                # Update labels and scores
                result_df.at[idx, 'insurance_label'] = delimiter.join(labels)
                result_df.at[idx, 'similarity_scores'] = delimiter.join(
                    [f"{label} ({score}*)" for label, score in zip(labels, score_strings)]
                )
                
                # Mark classification method
                result_df.at[idx, 'classification_method'] = 'sector_based'
                
                stats['new'] += 1
        else:
            # Already classified company, but check if we can improve
            current_labels = row['insurance_label'].split(delimiter)
            
            # If it has a low similarity score, try to improve
            min_score = 0.0
            if pd.notna(row['similarity_scores']) and row['similarity_scores'] != '':
                # Extract scores from formatted string
                try:
                    score_texts = row['similarity_scores'].split(delimiter)
                    scores = [float(st.split('(')[1].split(')')[0]) for st in score_texts]
                    if scores:
                        min_score = min(scores)
                except Exception as e:
                    logger.warning(f"Error parsing scores: {e}")
            
            # If minimum score is below desired threshold, try to improve
            if min_score < 0.4:  # Threshold to decide when to replace
                new_labels, new_scores = predict_by_sector(row, mappings, top_k)
                
                if new_labels and new_scores[0] >= min_confidence and new_scores[0] > min_score:
                    # Convert scores to formatted strings
                    score_strings = [f"{score:.2f}" for score in new_scores]
                    
                    # Update labels and scores
                    result_df.at[idx, 'insurance_label'] = delimiter.join(new_labels)
                    result_df.at[idx, 'similarity_scores'] = delimiter.join(
                        [f"{label} ({score}*)" for label, score in zip(new_labels, score_strings)]
                    )
                    
                    # Marchează metoda de clasificare
                    result_df.at[idx, 'classification_method'] = 'sector_based'
                    
                    stats['improved'] += 1
        
        stats['total'] += 1
        
    # Calculate new coverage
    labeled_mask = result_df['insurance_label'].notna() & (result_df['insurance_label'] != '')
    labeled_count = labeled_mask.sum()
    coverage_percentage = labeled_count / len(result_df) * 100
    
    logger.info(f"Sector-based classification completed:")
    logger.info(f"Companies processed: {stats['total']}")
    logger.info(f"Newly labeled: {stats['new']}")
    logger.info(f"Classifications improved: {stats['improved']}")
    logger.info(f"Final coverage: {labeled_count}/{len(result_df)} companies "
               f"({coverage_percentage:.2f}%)")
    
    return result_df

def generate_final_report(final_df: pd.DataFrame, 
                      initial_df: pd.DataFrame, 
                      config: Dict,
                      taxonomy_labels: List[str],
                      logger: logging.Logger,
                      model_components: Dict = None) -> None:
    """
    Generate final report and visualizations.
    
    Args:
        final_df: Final DataFrame with classifications
        initial_df: Initial DataFrame before all improvements
        config: Configuration dictionary
        taxonomy_labels: List of taxonomy labels
        logger: Logger instance
        model_components: Optional model components for taxonomy embeddings
    """
    results_dir = config.get("paths", {}).get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Calculate coverage statistics
    initial_labeled = initial_df['insurance_label'].notna() & (initial_df['insurance_label'] != '')
    final_labeled = final_df['insurance_label'].notna() & (final_df['insurance_label'] != '')
    
    initial_coverage = initial_labeled.sum() / len(initial_df) * 100
    final_coverage = final_labeled.sum() / len(final_df) * 100
    
    # Calculate label distribution
    def extract_all_labels(df):
        all_labels = []
        for labels in df.loc[df['insurance_label'].notna(), 'insurance_label']:
            if isinstance(labels, str) and labels:
                all_labels.extend(labels.split('; '))
        return all_labels
    
    initial_label_counts = Counter(extract_all_labels(initial_df))
    final_label_counts = Counter(extract_all_labels(final_df))
    
    # Generate report
    report_file = os.path.join(results_dir, "classification_report.txt")
    with open(report_file, 'w') as f:
        f.write("Insurance Taxonomy Classification Report\n")
        f.write("=====================================\n\n")
        
        f.write("Coverage Statistics:\n")
        f.write(f"- Initial coverage: {initial_labeled.sum()}/{len(initial_df)} companies "
                f"({initial_coverage:.2f}%)\n")
        f.write(f"- Final coverage: {final_labeled.sum()}/{len(final_df)} companies "
                f"({final_coverage:.2f}%)\n")
        f.write(f"- Improvement: {final_labeled.sum() - initial_labeled.sum()} additional companies "
                f"({final_coverage - initial_coverage:.2f}% increase)\n\n")
        
        f.write("Label Distribution (Top 20):\n")
        for label, count in final_label_counts.most_common(20):
            initial_count = initial_label_counts.get(label, 0)
            f.write(f"- {label}: {count} companies (was {initial_count})\n")
        
        f.write("\n\nSector-based Coverage:\n")
        for sector in final_df['sector'].dropna().unique():
            sector_df = final_df[final_df['sector'] == sector]
            sector_labeled = sector_df['insurance_label'].notna() & (sector_df['insurance_label'] != '')
            sector_coverage = sector_labeled.sum() / len(sector_df) * 100
            f.write(f"- {sector}: {sector_labeled.sum()}/{len(sector_df)} companies "
                    f"({sector_coverage:.2f}%)\n")
    
    logger.info(f"Report generated and saved to {report_file}")
    
    # Generate evaluation report and dashboard
    try:
        final_df['insurance_label'] = final_df['insurance_label'].fillna('')
        
        matched_labels = []
        for label_str in final_df['insurance_label']:
            if isinstance(label_str, str) and label_str.strip():
                matched_labels.append(label_str.split('; '))
            else:
                matched_labels.append([])
        
        # Create mock prediction results
        mock_prediction_results = {
            'matched_labels': matched_labels,
            'taxonomy_labels': taxonomy_labels
        }
        
        # Add taxonomy embeddings if available
        if model_components and "taxonomy_embeddings" in model_components:
            mock_prediction_results['taxonomy_embeddings'] = model_components["taxonomy_embeddings"]
        
        # Generate visualizations
        generate_evaluation_report(final_df, mock_prediction_results, results_dir)
        generate_report_dashboard(final_df, mock_prediction_results, 
                                os.path.join(results_dir, "dashboard"), 
                                taxonomy_labels)
        logger.info("Evaluation reports and dashboard generated")
    except Exception as e:
        logger.error(f"Could not generate visualization reports: {e}")
        logger.exception(e)

def main():
    """Main function to run the unified classification workflow."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Unified Insurance Taxonomy Classification")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                      help="Path to configuration file")
    parser.add_argument("--log_level", type=str, default="INFO",
                      choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                      help="Logging level")
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging(args.log_level)
    
    # Load configuration
    config = load_config(args.config)
    
    # Create results directory
    results_dir = config.get("paths", {}).get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Step 1: Train initial model
    logger.info("Step 1: Training initial model")
    model_components = train_initial_model(config, logger)
    
    # Step 2: Perform initial classification
    logger.info("Step 2: Performing initial classification")
    initial_results = initial_classification(model_components, config, logger)
    initial_df = initial_results["output_df"]
    
    # Step 3: Reclassify unlabeled companies with lower thresholds
    logger.info("Step 3: Reclassifying unlabeled companies")
    text_columns = config.get("data", {}).get("text_columns", ["description", "business_tags"])
    reclassified_df = reclassify_unlabeled(
        initial_df, 
        model_components["classifier"],
        model_components["embedding_generator"],
        text_columns,
        logger
    )
    
    # Save intermediate results
    intermediate_file = os.path.join(results_dir, "classified_companies_reclassified.csv")
    reclassified_df.to_csv(intermediate_file, index=False)
    logger.info(f"Intermediate results saved to {intermediate_file}")
    
    # Step 4: Apply sector-based classification
    logger.info("Step 4: Applying sector-based classification")
    final_df = sector_based_classification(
        reclassified_df, 
        min_confidence=0.3,
        top_k=3,
        logger=logger
    )
    
    # Save final results
    final_file = os.path.join(results_dir, "classified_companies_final.csv")
    final_df.to_csv(final_file, index=False)
    logger.info(f"Final results saved to {final_file}")
    
    # Step 5: Generate final report
    logger.info("Step 5: Generating final report")
    generate_final_report(
        final_df,
        initial_df,
        config,
        model_components["taxonomy_labels"],
        logger,
        model_components
    )
    
    logger.info("Classification workflow completed successfully")

if __name__ == "__main__":
    main()