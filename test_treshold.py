import logging
import pandas as pd
import numpy as np
import pickle
import os

# Configurare logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Verificăm dacă există clasificator salvat
model_path = 'models/classifier.pkl'
if os.path.exists(model_path):
    logger.info(f"Încărcare clasificator din {model_path}")
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    
    # Afișăm configurația salvată
    logger.info(f"Configurația salvată a clasificatorului:")
    logger.info(f"- similarity_threshold: {data.get('similarity_threshold', 'N/A')}")
    logger.info(f"- top_k_labels: {data.get('top_k_labels', 'N/A')}")
    logger.info(f"- min_similarity_score: {data.get('min_similarity_score', 'N/A')}")
else:
    logger.warning(f"Nu s-a găsit clasificator salvat la {model_path}")

# Căutăm fișierele în mai multe locații posibile
def find_file(file_name, possible_paths):
    for path in possible_paths:
        full_path = os.path.join(path, file_name)
        if os.path.exists(full_path):
            logger.info(f"Fișier găsit: {full_path}")
            return full_path
    logger.error(f"Nu s-a găsit fișierul: {file_name}")
    return None

# Locații posibile pentru fișiere
possible_paths = [
    '.', 
    'data', 
    'data/raw', 
    'data/processed',
    '../data',
    '../data/raw',
    '../data/processed'
]

# Găsim calea către fișierele necesare
taxonomy_file = find_file('insurance_taxonomy.csv', possible_paths)
companies_file = find_file('ml_insurance_challenge.csv', possible_paths)

if not taxonomy_file or not companies_file:
    logger.error("Nu s-au găsit fișierele necesare. Se oprește execuția.")
    exit(1)

# Rulăm un test de predicție manual pentru a verifica procesul
from src.models.classifier import InsuranceTaxonomyClassifier
from src.features.embeddings import EmbeddingGenerator

# Încărcăm taxonomia
logger.info("Încărcare taxonomie...")
taxonomy_df = pd.read_csv(taxonomy_file)
taxonomy_labels = taxonomy_df['label'].tolist()
logger.info(f"Taxonomie încărcată: {len(taxonomy_labels)} etichete")

# Creăm un generator de embeddings
logger.info("Inițializare generator de embeddings...")
embedding_gen = EmbeddingGenerator(model_name="all-MiniLM-L6-v2", max_seq_length=128)

# Generăm embeddings pentru taxonomie
logger.info("Generare embeddings pentru taxonomie...")
taxonomy_embeddings = embedding_gen.generate_embeddings(taxonomy_labels)

# Creăm și testăm clasificatori cu praguri diferite
test_thresholds = [0.5, 0.4, 0.35, 0.3]
for threshold in test_thresholds:
    logger.info(f"\nTest cu threshold = {threshold}")
    
    # Creăm clasificator cu pragul specificat
    classifier = InsuranceTaxonomyClassifier(
        similarity_threshold=threshold,
        top_k_labels=5,
        min_similarity_score=threshold - 0.1  # Un pic mai jos decât threshold-ul principal
    )
    
    # Configurăm taxonomia
    classifier.taxonomy_labels = taxonomy_labels
    classifier.taxonomy_embeddings = taxonomy_embeddings
    
    # Testăm pe câteva exemple simple
    test_texts = [
        "A construction company that builds commercial buildings and infrastructure.",
        "An insurance broker that helps businesses find property insurance.",
        "A small bakery that sells bread and pastries.",
        "A technology company developing software for financial institutions."
    ]
    
    # Generăm embeddings pentru textele de test
    test_embeddings = embedding_gen.generate_embeddings(test_texts)
    
    # Facem predicții
    results = classifier.predict(test_embeddings)
    
    # Afișăm rezultatele
    for i, (text, labels, scores) in enumerate(zip(
            test_texts, 
            results["matched_labels"], 
            results["similarity_scores"])):
        logger.info(f"Text {i+1}: {text[:50]}...")
        if labels:
            for label, score in zip(labels, scores):
                logger.info(f"  - {label}: {score:.4f}")
        else:
            logger.info("  - Nicio etichetă găsită")
    
    # Verificăm numărul de rezultate per threshold
    has_labels = sum(1 for labels in results["matched_labels"] if labels)
    logger.info(f"Texte cu etichete: {has_labels}/{len(test_texts)} ({has_labels/len(test_texts)*100:.0f}%)")