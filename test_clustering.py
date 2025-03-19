#!/usr/bin/env python3
"""
Test script for clustering of company embeddings.
"""
import logging
import pandas as pd
import numpy as np
import os
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.preprocessing import TextPreprocessor, load_data, prepare_company_features, prepare_taxonomy_features
from src.features.embeddings import EmbeddingGenerator
from src.models.clustering import EmbeddingClusterer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Test the clustering on a small dataset."""
    try:
        # Load the data
        logger.info("Loading data...")
        companies_df, taxonomy_df = load_data(
            "ml_insurance_challenge.csv",
            "insurance_taxonomy.csv"
        )
        
        # Print data overview
        logger.info(f"Loaded {len(companies_df)} companies and {len(taxonomy_df)} taxonomy labels")
        
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
        
        # Sample test with a subset for faster testing
        logger.info("Creating a small test subset...")
        test_companies = processed_companies.head(500).copy()
        
        # Initialize embedding generator
        logger.info("Initializing embedding generator...")
        embedding_config = {
            'model_name': 'all-MiniLM-L6-v2',
            'max_seq_length': 128
        }
        embedding_gen = EmbeddingGenerator(embedding_config)
        
        # Generate embeddings for companies
        logger.info("Generating embeddings for companies...")
        company_embeddings = embedding_gen.generate_embeddings(
            test_companies['combined_text'].tolist()
        )
        
        # Initialize clusterer
        logger.info("Initializing clusterer...")
        clustering_config = {
            'algorithm': 'kmeans',
            'n_clusters': 10  # Small number for test
        }
        clusterer = EmbeddingClusterer(clustering_config)
        
        # Fit the clustering model
        logger.info("Fitting clustering model...")
        clusterer.fit(company_embeddings)
        
        # Get cluster labels
        cluster_labels = clusterer.get_cluster_labels()
        
        # Add cluster labels to the companies DataFrame
        test_companies['cluster'] = cluster_labels
        
        # Show cluster distribution
        cluster_counts = test_companies['cluster'].value_counts().sort_index()
        print("\nCluster distribution:")
        for cluster, count in cluster_counts.items():
            print(f"Cluster {cluster}: {count} companies")
        
        # Visualize clusters
        logger.info("Visualizing clusters...")
        fig, tsne_embeddings = clusterer.visualize_clusters(company_embeddings)
        
        # Save the visualization
        plt.tight_layout()
        plt.savefig('cluster_visualization.png')
        logger.info("Saved cluster visualization to 'cluster_visualization.png'")
        
        # Analyze clusters
        logger.info("Analyzing clusters...")
        cluster_analysis = clusterer.analyze_clusters(test_companies)
        
        # Show cluster analysis
        print("\nCluster analysis:")
        for _, row in cluster_analysis.iterrows():
            print(f"\nCluster {row['cluster']} (Size: {row['size']})")
            print(f"Top terms: {row['top_terms']}")
        
        # Show some examples from each cluster
        print("\nExample companies from each cluster:")
        for cluster in range(clustering_config['n_clusters']):
            cluster_examples = test_companies[test_companies['cluster'] == cluster].head(2)
            print(f"\nCluster {cluster}:")
            for i, row in cluster_examples.iterrows():
                print(f"  - {row['description'][:100]}...")
        
        logger.info("Test completed successfully")
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()