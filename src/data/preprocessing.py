"""
Preprocessing module.

This module implements text preprocessing functionalities for cleaning and preparing
text data before generating embeddings.
"""

import re
import string
import logging
from typing import List, Dict, Optional, Union

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)

# Ensure required NLTK resources are downloaded
def download_nltk_resources():
    """Download required NLTK resources if they don't exist."""
    resources = ['punkt', 'wordnet', 'stopwords', 'omw-1.4']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else resource)
        except LookupError:
            nltk.download(resource, quiet=True)

# Call this at module load time
download_nltk_resources()

class TextPreprocessor:
    """
    Class for preprocessing text data.
    """
    
    def __init__(self, 
                 remove_stopwords: bool = True,
                 min_word_length: int = 2,
                 lemmatize: bool = True,
                 language: str = 'english'):
        """
        Initialize the text preprocessor.
        
        Args:
            remove_stopwords: Whether to remove stopwords
            min_word_length: Minimum length of words to keep
            lemmatize: Whether to apply lemmatization
            language: Language for stopwords
        """
        self.remove_stopwords = remove_stopwords
        self.min_word_length = min_word_length
        self.lemmatize = lemmatize
        
        # Initialize lemmatizer if lemmatization is enabled
        if lemmatize:
            self.lemmatizer = WordNetLemmatizer()
            
        # Load stopwords if stopword removal is enabled
        if remove_stopwords:
            self.stop_words = set(stopwords.words(language))
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess a single text.
        
        Args:
            text: Text to preprocess
            
        Returns:
            Preprocessed text
        """
        if not isinstance(text, str) or not text.strip():
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Filter out short words
        tokens = [token for token in tokens if len(token) >= self.min_word_length]
        
        # Remove stopwords if enabled
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
            
        # Apply lemmatization if enabled
        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
            
        # Join tokens back into a single string
        return ' '.join(tokens)
    
    def preprocess_texts(self, texts: List[str]) -> List[str]:
        """
        Preprocess a list of texts.
        
        Args:
            texts: List of texts to preprocess
            
        Returns:
            List of preprocessed texts
        """
        return [self.preprocess_text(text) for text in texts]
    
    def preprocess_dataframe(self, df: pd.DataFrame, 
                           text_columns: Union[str, List[str]],
                           inplace: bool = False) -> pd.DataFrame:
        """
        Preprocess text columns in a DataFrame.
        
        Args:
            df: DataFrame containing the text columns
            text_columns: Column name or list of column names to preprocess
            inplace: Whether to modify the DataFrame in place
            
        Returns:
            DataFrame with preprocessed text columns
        """
        if not inplace:
            df = df.copy()
            
        # Convert single column to list
        if isinstance(text_columns, str):
            text_columns = [text_columns]
            
        # Preprocess each column
        for col in text_columns:
            if col in df.columns:
                logger.info(f"Preprocessing column: {col}")
                df[f"{col}_processed"] = df[col].apply(self.preprocess_text)
            else:
                logger.warning(f"Column {col} not found in DataFrame")
                
        return df
    
    def preprocess_taxonomy(self, taxonomy: List[str]) -> List[str]:
        """
        Preprocess taxonomy labels.
        
        Args:
            taxonomy: List of taxonomy labels
            
        Returns:
            List of preprocessed taxonomy labels
        """
        logger.info("Preprocessing taxonomy labels")
        return self.preprocess_texts(taxonomy)


def clean_business_tags(tags_string: str) -> List[str]:
    """
    Clean and extract business tags from a string.
    
    Args:
        tags_string: String containing business tags, typically comma separated
        
    Returns:
        List of cleaned business tags
    """
    if not isinstance(tags_string, str) or not tags_string.strip():
        return []
        
    # Split by comma
    tags = tags_string.split(',')
    
    # Strip whitespace
    tags = [tag.strip() for tag in tags]
    
    # Remove empty tags
    tags = [tag for tag in tags if tag]
    
    return tags


def load_and_preprocess_data(company_file: str, 
                           taxonomy_file: str,
                           preprocessor: Optional[TextPreprocessor] = None) -> Dict:
    """
    Load and preprocess company and taxonomy data.
    
    Args:
        company_file: Path to the company data file
        taxonomy_file: Path to the taxonomy file
        preprocessor: TextPreprocessor instance (creates one if None)
        
    Returns:
        Dictionary containing preprocessed data
    """
    logger.info(f"Loading data from {company_file} and {taxonomy_file}")
    
    # Load company data
    company_df = pd.read_csv(company_file)
    
    # Load taxonomy data
    taxonomy_df = pd.read_csv(taxonomy_file)
    taxonomy_labels = taxonomy_df['label'].tolist()
    
    # Create preprocessor if not provided
    if preprocessor is None:
        preprocessor = TextPreprocessor()
    
    # Preprocess company data
    company_df = preprocessor.preprocess_dataframe(company_df, ['description', 'business_tags'])
    
    # Preprocess taxonomy labels
    preprocessed_taxonomy = preprocessor.preprocess_taxonomy(taxonomy_labels)
    
    # Clean business tags
    company_df['business_tags_list'] = company_df['business_tags'].apply(clean_business_tags)
    
    # Return preprocessed data
    return {
        'company_df': company_df,
        'taxonomy_labels': taxonomy_labels,
        'preprocessed_taxonomy': preprocessed_taxonomy
    }