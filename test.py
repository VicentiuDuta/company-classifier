import pandas as pd
from src.models.classifier import InsuranceTaxonomyClassifier
from src.features.embeddings import EmbeddingGenerator

# 1. Load the pre-trained classifier
classifier = InsuranceTaxonomyClassifier.load('models/classifier.pkl')

# 2. Create an embedding generator (use the same model as during training)
embedding_gen = EmbeddingGenerator(
    model_name="all-MiniLM-L6-v2", 
    max_seq_length=128
)

# 3. Prepare some test texts to classify
test_texts = [
    "A construction company specializing in commercial building projects",
    "An insurance broker providing property and liability coverage",
    "A technology startup developing financial software solutions"
]

# 4. Predict labels for the texts
prediction_results = classifier.predict_for_texts(
    embedding_gen, 
    test_texts
)

# 5. Print the results
for text, labels, scores in zip(
    test_texts, 
    prediction_results['matched_labels'], 
    prediction_results['similarity_scores']
):
    print(f"\nText: {text}")
    if labels:
        print("Predicted Labels:")
        for label, score in zip(labels, scores):
            print(f"  - {label} (Similarity: {score:.2f})")
    else:
        print("  No labels found")

# 6. You can also classify an entire DataFrame
# Load your companies data
df = pd.read_csv('data/raw/ml_insurance_challenge.csv')

# Combine text fields
combined_texts = embedding_gen.combine_text_fields(
    df, 
    ['description', 'business_tags'], 
    weights=[1.0, 0.5]
)

# Generate predictions
df_predictions = classifier.predict(
    embedding_gen.generate_embeddings(combined_texts)
)

# Create output DataFrame with labels
results_df = classifier.create_output_dataframe(
    df, 
    df_predictions
)

# Save results
results_df.to_csv('results/classified_companies.csv', index=False)
