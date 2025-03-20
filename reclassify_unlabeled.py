import logging
import pandas as pd
import os
from src.models.classifier import InsuranceTaxonomyClassifier
from src.features.embeddings import EmbeddingGenerator

# Configurare logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Încarcă datele clasificate existente
logger.info("Încărcare date clasificate...")
results_file = 'results/classified_companies.csv'
if not os.path.exists(results_file):
    logger.error(f"Fișierul {results_file} nu există!")
    exit(1)

df = pd.read_csv(results_file)
logger.info(f"Date încărcate: {len(df)} companii")

# Identifică companiile fără etichete
unlabeled_mask = df['insurance_label'].isna() | (df['insurance_label'] == '')
unlabeled_companies = df[unlabeled_mask]
logger.info(f"Companii fără etichete: {len(unlabeled_companies)} din {len(df)} ({len(unlabeled_companies)/len(df)*100:.2f}%)")

# Încarcă clasificatorul
logger.info("Încărcare clasificator...")
classifier = InsuranceTaxonomyClassifier.load('models/classifier.pkl')

# Setează un prag mai mic pentru a include mai multe companii
original_threshold = classifier.similarity_threshold
classifier.similarity_threshold = 0.30  # Un prag foarte mic pentru a maximiza acoperirea
classifier.min_similarity_score = 0.20  # Un prag minim mai mic
classifier.top_k_labels = 5  # Creșterea numărului maxim de etichete

logger.info(f"Prag de similaritate original: {original_threshold}, nou: {classifier.similarity_threshold}")
logger.info(f"Prag minim original: {classifier.min_similarity_score}, top_k: {classifier.top_k_labels}")

# Creează un generator de embeddings
logger.info("Inițializare generator de embeddings...")
embedding_gen = EmbeddingGenerator(model_name="all-MiniLM-L6-v2", max_seq_length=128)

# Verifică dacă există companii fără etichete
if len(unlabeled_companies) == 0:
    logger.info("Nu există companii fără etichete! Procesul se oprește.")
    exit(0)

# Pregătește textele pentru companiile neclasificate
logger.info("Pregătire texte pentru companiile neclasificate...")
text_columns = ['description', 'business_tags']
unlabeled_texts = embedding_gen.combine_text_fields(
    unlabeled_companies, text_columns, weights=[1.0, 0.5])

# Generează predicții pentru companiile neclasificate
logger.info("Generare predicții pentru companiile neclasificate...")
prediction_results = classifier.predict_for_texts(
    embedding_gen, unlabeled_texts)

# Creează un DataFrame cu predicțiile pentru companiile neclasificate
output_df = classifier.create_output_dataframe(unlabeled_companies, prediction_results)

# Verifică câte companii au primit etichete acum
newly_labeled = output_df['insurance_label'].str.len() > 0
newly_labeled_count = newly_labeled.sum()
logger.info(f"Companii noi clasificate: {newly_labeled_count} din {len(unlabeled_companies)} ({newly_labeled_count/len(unlabeled_companies)*100:.2f}%)")

# Actualizează DataFrame-ul original
unlabeled_indices = unlabeled_companies.index
df.loc[unlabeled_indices, 'insurance_label'] = output_df['insurance_label']
df.loc[unlabeled_indices, 'similarity_scores'] = output_df['similarity_scores']

# Salvează rezultatele actualizate
output_file = 'results/classified_companies_improved.csv'
df.to_csv(output_file, index=False)
logger.info(f"Rezultate actualizate salvate în {output_file}")

# Afișează statistici finale
total_labeled = df['insurance_label'].notna() & (df['insurance_label'] != '')
total_labeled_count = total_labeled.sum()
logger.info(f"Total companii clasificate după actualizare: {total_labeled_count} din {len(df)} ({total_labeled_count/len(df)*100:.2f}%)")

# Afișează câteva exemple de companii nou clasificate
if newly_labeled_count > 0:
    logger.info("\nExemple de companii nou clasificate:")
    newly_labeled_examples = output_df[newly_labeled].head(5)
    for i, row in newly_labeled_examples.iterrows():
        logger.info(f"\nCompanie: {row['description'][:100]}...")
        logger.info(f"Etichete atribuite: {row['insurance_label']}")
        logger.info(f"Scoruri: {row['similarity_scores']}")
