
# Insurance Taxonomy Classification: Solution Presentation

## 1. Introduction and Problem Overview

The challenge was to build a robust company classifier for a new insurance taxonomy. Given:
- A dataset of 9,494 companies with descriptions, business tags, and sector/category/niche classifications
- A static taxonomy of 220 insurance industry labels
- No predefined ground truth for validation

The goal was to develop a solution that could accurately classify each company into one or more relevant insurance taxonomy labels, demonstrating effectiveness and explaining the approach's strengths and limitations.

## 2. Solution Architecture

I developed a multi-stage classification pipeline with three distinct phases, each addressing different aspects of the classification challenge:

### 2.1 Initial Semantic Classification

The first stage leverages semantic similarity between company descriptions and taxonomy labels:

1. **Text Preprocessing**: Cleaned and normalized company descriptions and business tags using:
   - Lemmatization
   - Stopword removal
   - Special character handling
   - Tokenization

2. **Embedding Generation**: Used the SentenceTransformer model `all-MiniLM-L6-v2` to convert:
   - Company texts into dense vector representations (embeddings)
   - Insurance taxonomy labels into the same vector space

3. **Similarity-Based Matching**: Calculated cosine similarity between company and taxonomy embeddings, applying configurable thresholds to determine matches.

### 2.2 Threshold Relaxation for Low-Confidence Companies

For companies that received no labels or very low confidence scores in the first stage:

1. **Identified Unlabeled Companies**: Selected companies with empty label assignments  
2. **Lowered Classification Thresholds**: Reduced similarity threshold from `0.35` to `0.30`  
3. **Expanded Top-K Selection**: Increased maximum number of potential labels from `3` to `5`  
4. **Applied More Lenient Matching**: Reclassified using these relaxed parameters  

### 2.3 Sector-Based Classification Fallback

For any remaining unclassified companies, implemented a domain knowledge-based approach:

1. **Built Sector-to-Label Mappings**: Created association maps between company sectors/categories/niches and insurance labels based on already classified companies  
2. **Weighted Label Assignment**: Assigned scores to potential labels based on their frequency in similar sectors  
3. **Provided Complete Coverage**: Ensured every company received at least one relevant label  

## 3. Technical Implementation

### 3.1 Data Preprocessing Module

The preprocessing module (`src/data/preprocessing.py`) implements:
- A `TextPreprocessor` class with configurable preprocessing options  
- Business tag extraction and cleaning  
- Combined preprocessing pipeline for company and taxonomy data  

```python
def preprocess_text(self, text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if len(token) >= self.min_word_length]

    if self.remove_stopwords:
        tokens = [token for token in tokens if token not in self.stop_words]

    if self.lemmatize:
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]

    return ' '.join(tokens)
```

### 3.2 Embedding Generation

The embedding module (`src/features/embeddings.py`) handles:
- Loading and configuring the SentenceTransformer model  
- Batch generation of embeddings to handle large datasets efficiently  
- Contextual text combination to enhance semantic representation  

```python
def combine_text_fields(self, df: pd.DataFrame, text_columns: List[str], weights: Optional[List[float]] = None) -> List[str]:
    if weights is None:
        weights = [1.0] * len(text_columns)

    combined_texts = []

    for _, row in df.iterrows():
        context_parts = []

        if 'sector' in df.columns and pd.notna(row['sector']):
            context_parts.append(f"Sector: {row['sector']}")
        if 'category' in df.columns and pd.notna(row['category']):
            context_parts.append(f"Category: {row['category']}")
        if 'niche' in df.columns and pd.notna(row['niche']):
            context_parts.append(f"Niche: {row['niche']}")

        text_parts = []
        for col, weight in zip(text_columns, weights):
            if pd.notna(row[col]) and row[col]:
                text = str(row[col])
                text = " ".join([text] * int(weight))
                text_parts.append(text)

        full_text = " ".join(context_parts + text_parts)
        combined_texts.append(full_text)

    return combined_texts
```

### 3.3 Classification Logic

The core classifier (`src/models/classifier.py`) implements:
- Configurable similarity thresholds and selection parameters  
- Efficient batch processing for large datasets  
- Output formatting and score tracking  

```python
def _predict_batch(self, query_embeddings: np.ndarray) -> Dict:
    query_norm = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    taxonomy_norm = self.taxonomy_embeddings / np.linalg.norm(self.taxonomy_embeddings, axis=1, keepdims=True)

    similarity_matrix = np.dot(query_norm, taxonomy_norm.T)

    batch_results = {
        'matched_labels': [],
        'similarity_scores': []
    }

    for i in range(len(query_embeddings)):
        scores = similarity_matrix[i]
        matches = np.where(scores >= self.min_similarity_score)[0]
        sorted_indices = matches[np.argsort(scores[matches])[::-1]]
        top_indices = sorted_indices[:self.top_k_labels]

        query_labels = [self.taxonomy_labels[idx] for idx in top_indices]
        query_scores = scores[top_indices].tolist()

        final_labels = []
        final_scores = []

        for label, score in zip(query_labels, query_scores):
            if score >= self.similarity_threshold:
                final_labels.append(label)
                final_scores.append(score)

        batch_results['matched_labels'].append(final_labels)
        batch_results['similarity_scores'].append(final_scores)

    return batch_results
```

### 3.4 Sector-Based Classification

For the domain knowledge component, implemented mapping and fallback classification:

```python
def build_sector_mappings(df: pd.DataFrame, label_column: str = 'insurance_label', delimiter: str = '; ', min_count: int = 2) -> Dict:
    labeled_df = df[df[label_column].notna() & (df[label_column] != '')]

    sector_to_labels = defaultdict(Counter)
    category_to_labels = defaultdict(Counter)
    niche_to_labels = defaultdict(Counter)

    for _, row in labeled_df.iterrows():
        if pd.isna(row[label_column]) or row[label_column] == '':
            continue

        labels = row[label_column].split(delimiter)

        if pd.notna(row.get('sector', None)):
            for label in labels:
                sector_to_labels[row['sector']][label] += 1

        if pd.notna(row.get('category', None)):
            for label in labels:
                category_to_labels[row['category']][label] += 1

        if pd.notna(row.get('niche', None)):
            for label in labels:
                niche_to_labels[row['niche']][label] += 1

    for sector in sector_to_labels:
        sector_to_labels[sector] = {k: v for k, v in sector_to_labels[sector].items() if v >= min_count}

    return {
        'sector_to_labels': dict(sector_to_labels),
        'category_to_labels': dict(category_to_labels),
        'niche_to_labels': dict(niche_to_labels)
    }
```

## 4. Results and Performance

### 4.1 Classification Coverage

| Stage                        | Companies | Coverage   |
|------------------------------|-----------|------------|
| Initial Semantic Classification | 7,400     | 77.94%     |
| After Threshold Relaxation   | 9,250     | 97.43%     |
| Sector-Based Classification  | 9,494     | 100.00%    |

### 4.2 Label Distribution (Top 10)

| Insurance Label              | Count | Percentage |
|------------------------------|-------|------------|
| Apparel Manufacturing        | 1,473 | 15.5%      |
| Marketing Services           | 1,382 | 14.6%      |
| Gas Manufacturing Services   | 1,248 | 13.1%      |
| Accessory Manufacturing      | 1,193 | 12.6%      |
| Travel Services              | 1,078 | 11.4%      |
| Real Estate Services         | 915   | 9.6%       |
| Livestock Dealer Services    | 647   | 6.8%       |
| Agricultural Equipment Services | 588 | 6.2%       |
| Cabinetry Manufacturing      | 558   | 5.9%       |
| Furniture Manufacturing      | 411   | 4.3%       |

### 4.3 Sector-Based Coverage

| Sector        | Companies | Coverage |
|---------------|-----------|----------|
| Services      | 3,556     | 100.00%  |
| Manufacturing | 4,005     | 100.00%  |
| Wholesale     | 779       | 100.00%  |
| Retail        | 571       | 100.00%  |
| Government    | 255       | 100.00%  |
| Non Profit    | 140       | 100.00%  |
| Education     | 161       | 100.00%  |

### 4.4 Classification Method Distribution

| Classification Method  | Count | Percentage |
|------------------------|-------|------------|
| Semantic               | 7,400 | 77.94%     |
| Threshold Relaxation   | 1,850 | 19.49%     |
| Sector-Based           | 244   | 2.57%      |

## 5. Analysis of Approach

### 5.1 Strengths

- **High Coverage**: 100% classification achieved  
- **Semantic Understanding**: Captures nuanced relationships  
- **Confidence Scores**: Threshold-based filtering  
- **Robust to Missing Data**: Fallback mechanisms  
- **Scalable Architecture**: Handles large datasets  
- **Domain Knowledge Integration**: Leverages industry patterns  

### 5.2 Limitations

- **Quality vs. Coverage Trade-off**  
- **Model Dependency**  
- **Threshold Sensitivity**  
- **Label Imbalance**  
- **Limited Validation**  

## 6. Scalability and Future Improvements

### 6.1 Scalability Considerations

- Batch processing  
- Vectorized similarity calculations  
- Incremental updates and fast classification  

### 6.2 Future Improvements

- **Embedding Enhancements**  
- **Active Learning Integration**  
- **Advanced Ensemble Methods**  
- **Hierarchical Classification**  
- **Explainability Improvements**  

## 7. Conclusion

The multi-stage classification approach successfully addresses the challenge of mapping companies to an insurance taxonomy without labeled training data. By combining semantic understanding with domain knowledge:
- Achieved **complete coverage**
- Balanced **precision and recall**
- **Scalable and adaptable** architecture  
- Ready for **future enhancements** and **human-in-the-loop** validation  
