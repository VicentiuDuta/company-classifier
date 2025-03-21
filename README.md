
# Insurance Taxonomy Classifier

## 📄 Description
This project implements a robust classifier for companies in the insurance industry, based on a static taxonomy. It leverages advanced natural language processing (NLP) techniques to automatically categorize companies using semantic embeddings and multi-stage classification strategies.

---

## 📁 Project Structure
```
insurance-taxonomy-classifier/
├── config/          # Configuration files
├── data/            # Raw and processed data files
├── results/         # Classification results and reports
├── src/             # Source code
│   ├── data/        # Data preprocessing modules
│   ├── features/    # Embedding generation
│   ├── models/      # Classification logic
│   └── utils/       # Reporting and visualization tools
└── tests/           # Unit tests and validation scripts
```

---

## ✨ Key Features
- **Semantic embedding-based classification**
- **Multi-stage classification approach**
- **Configurable similarity thresholds**
- **Sector-based classification fallback**
- **Comprehensive reporting and visualization**

---

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/VicentiuDuta/company-classifier.git
cd insurance-taxonomy-classifier

# Create a virtual environment
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

---

## ⚙️ Usage

### Test the classifier:
``` bash
python classify_company.py
```

### Run the classification process:
```bash
python main.py
```

### Optional: Customize configuration
```bash
python main.py --config config/config.yaml
```

---

## 🔄 Classification Workflow

The classification process involves **three main stages**:

### 1. Initial Embedding-Based Classification
- Generate semantic embeddings for companies and taxonomy labels
- Classify using initial similarity thresholds
- Achieve initial coverage (~77-80%)

### 2. Reclassification of Unlabeled Companies
- Lower similarity thresholds
- Attempt to classify companies missed in the first stage
- Increase coverage to ~90-95%

### 3. Sector-Based Classification
- Use domain knowledge and sector mappings
- Assign labels to remaining unclassified companies
- Aim for 100% coverage

---

## 📊 Key Model Files

### classifier.pkl
This file contains the serialized (saved) state of the trained classifier, created using Python's `pickle` library. It is automatically generated during the training process and saved in the `models/` directory.

**Contents:**
- Configured similarity thresholds for classification
- Maximum number of labels per company
- Minimum similarity score to consider a match
- Insurance taxonomy labels
- Pre-computed embeddings for these labels

**Benefits:**
- Eliminates the need to regenerate taxonomy embeddings for each run
- Enables standalone classification applications (like `classify_company.py`)
- Significantly reduces startup time when classifying new companies

**Usage:**
The classifier is loaded automatically by both the main pipeline and the interactive classification script:

```python
# Example of loading the classifier
from src.models.classifier import InsuranceTaxonomyClassifier
classifier = InsuranceTaxonomyClassifier.load("models/classifier.pkl")
```
If you modify the taxonomy or retrain the model, a new classifier.pkl file will be generated.

## 📦 Dependencies
- `sentence-transformers`
- `pandas`
- `numpy`
- `scikit-learn`
- `nltk`
- `matplotlib`
- `PyYAML`

---

## ⚙️ Configuration
The project uses a flexible **YAML configuration** system.  
Key configuration options include:
- Preprocessing parameters
- Embedding generation settings
- Classification thresholds
- Output paths

➡️ Customize `config/config.yaml` to adjust the classification process as needed.

---

## 🚧 Potential Improvements
- Integrate machine learning models for refinement  
- Develop more sophisticated embedding techniques  
- Implement active learning mechanisms  
- Enhance threshold optimization strategies  

---