# Insurance Taxonomy Classifier

## Description
This project implements a robust classifier for companies in the insurance industry, based on a static taxonomy.

## Project Structure
The project is organized into the following directories:
- `data/`: Contains raw and processed data
- `notebooks/`: Jupyter notebooks for exploration and analysis
- `src/`: Source code for the project
- `config/`: Configuration files
- `results/`: Results and visualizations
- `tests/`: Unit tests

## Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/insurance-taxonomy-classifier.git
cd insurance-taxonomy-classifier

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Usage
```bash
# Run the classification process
python main.py
```

## Approach
The project uses a hybrid approach that combines NLP, embeddings, and clustering algorithms to classify companies into categories from the insurance taxonomy.
