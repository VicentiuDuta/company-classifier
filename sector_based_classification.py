import logging
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import os

# Configurare logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def build_sector_mappings(df, label_column='insurance_label', delimiter='; ', min_count=2):
    """
    Construiește mapări între sectoare/categorii/nișe și etichetele de asigurare.
    
    Args:
        df: DataFrame cu companii clasificate
        label_column: Numele coloanei cu etichete
        delimiter: Delimitatorul pentru etichetele multiple
        min_count: Numărul minim de apariții pentru a include o etichetă
        
    Returns:
        Dicționar cu mapările
    """
    # Folosim doar companiile care au deja etichete
    labeled_df = df[df[label_column].notna() & (df[label_column] != '')]
    
    # Inițializare dicționare
    sector_to_labels = defaultdict(Counter)
    category_to_labels = defaultdict(Counter)
    niche_to_labels = defaultdict(Counter)
    
    # Construiește mapările
    for _, row in labeled_df.iterrows():
        if pd.isna(row[label_column]) or row[label_column] == '':
            continue
            
        labels = row[label_column].split(delimiter)
        
        # Mapare sector -> etichete
        if pd.notna(row.get('sector', None)):
            for label in labels:
                sector_to_labels[row['sector']][label] += 1
        
        # Mapare categorie -> etichete
        if pd.notna(row.get('category', None)):
            for label in labels:
                category_to_labels[row['category']][label] += 1
        
        # Mapare nișă -> etichete
        if pd.notna(row.get('niche', None)):
            for label in labels:
                niche_to_labels[row['niche']][label] += 1
    
    # Filtrează etichetele rare
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

def predict_by_sector(company, mappings, top_k=3):
    """
    Prezice etichetele pentru o companie pe baza sectorului, categoriei și nișei.
    
    Args:
        company: Rând din DataFrame reprezentând o companie
        mappings: Dicționar cu mapările construite
        top_k: Numărul maxim de etichete de returnat
        
    Returns:
        Lista de etichete prezise și scoruri
    """
    candidate_labels = Counter()
    
    # Adaugă etichete bazate pe sector
    sector = company.get('sector')
    if pd.notna(sector) and sector in mappings['sector_to_labels']:
        for label, count in mappings['sector_to_labels'][sector].items():
            candidate_labels[label] += count * 1  # Pondere standard pentru sector
    
    # Adaugă etichete bazate pe categorie
    category = company.get('category')
    if pd.notna(category) and category in mappings['category_to_labels']:
        for label, count in mappings['category_to_labels'][category].items():
            candidate_labels[label] += count * 2  # Pondere dublă pentru categorie (mai specifică)
    
    # Adaugă etichete bazate pe nișă
    niche = company.get('niche')
    if pd.notna(niche) and niche in mappings['niche_to_labels']:
        for label, count in mappings['niche_to_labels'][niche].items():
            candidate_labels[label] += count * 3  # Pondere triplă pentru nișă (cea mai specifică)
    
    # Returnează top k etichete
    top_labels = []
    top_scores = []
    
    for label, score in candidate_labels.most_common(top_k):
        top_labels.append(label)
        # Convertim scorul la o valoare între 0 și 1 pentru consistență cu clasificatorul bazat pe similaritate
        normalized_score = min(0.99, score / 10.0)  
        top_scores.append(normalized_score)
    
    return top_labels, top_scores

def apply_sector_classification(df, mappings, min_confidence=0.3, top_k=3, delimiter='; '):
    """
    Aplică clasificarea bazată pe sectoare pentru companiile neclasificate sau cu clasificări slabe.
    
    Args:
        df: DataFrame cu companii
        mappings: Dicționar cu mapările construite
        min_confidence: Scorul minim de încredere pentru a accepta o predicție
        top_k: Numărul maxim de etichete de returnat
        delimiter: Delimitatorul pentru etichetele multiple
        
    Returns:
        DataFrame actualizat
    """
    # Creează o copie pentru a nu modifica originalul
    result_df = df.copy()
    
    # Contoare pentru statistici
    stats = {'total': 0, 'improved': 0, 'new': 0}
    
    # Aplică clasificarea pentru fiecare companie
    for idx, row in result_df.iterrows():
        if pd.isna(row['insurance_label']) or row['insurance_label'] == '':
            # Companie neclasificată
            labels, scores = predict_by_sector(row, mappings, top_k)
            
            if labels and scores[0] >= min_confidence:
                # Convertim scorurile la string-uri formatate
                score_strings = [f"{score:.2f}" for score in scores]
                
                # Actualizăm etichetele și scorurile
                result_df.at[idx, 'insurance_label'] = delimiter.join(labels)
                result_df.at[idx, 'similarity_scores'] = delimiter.join(
                    [f"{label} ({score})" for label, score in zip(labels, score_strings)]
                )
                
                stats['new'] += 1
        else:
            # Companie deja clasificată, dar verificăm dacă putem îmbunătăți
            current_labels = row['insurance_label'].split(delimiter)
            
            # Dacă are un scor de similaritate mic, încercăm să îmbunătățim
            min_score = 0.0
            if pd.notna(row['similarity_scores']) and row['similarity_scores'] != '':
                # Extrage scorurile din string-ul formatat
                try:
                    score_texts = row['similarity_scores'].split(delimiter)
                    scores = [float(st.split('(')[1].split(')')[0]) for st in score_texts]
                    if scores:
                        min_score = min(scores)
                except Exception as e:
                    logger.warning(f"Eroare la parsarea scorurilor: {e}")
            
            # Dacă scorul minim este sub pragul dorit, încercăm să îmbunătățim
            if min_score < 0.4:  # Un prag pentru a decide când să înlocuim
                new_labels, new_scores = predict_by_sector(row, mappings, top_k)
                
                if new_labels and new_scores[0] >= min_confidence and new_scores[0] > min_score:
                    # Convertim scorurile la string-uri formatate
                    score_strings = [f"{score:.2f}" for score in new_scores]
                    
                    # Actualizăm etichetele și scorurile
                    result_df.at[idx, 'insurance_label'] = delimiter.join(new_labels)
                    result_df.at[idx, 'similarity_scores'] = delimiter.join(
                        [f"{label} ({score})" for label, score in zip(new_labels, score_strings)]
                    )
                    
                    stats['improved'] += 1
        
        stats['total'] += 1
        
        # Afișăm progresul la fiecare 1000 de companii
        if stats['total'] % 1000 == 0:
            logger.info(f"Procesate {stats['total']} companii, clasificate nou: {stats['new']}, îmbunătățite: {stats['improved']}")
    
    return result_df, stats

def main():
    # Încarcă datele clasificate îmbunătățite
    logger.info("Încărcare date clasificate...")
    input_file = 'results/classified_companies_improved.csv'
    if not os.path.exists(input_file):
        logger.error(f"Fișierul {input_file} nu există!")
        exit(1)
    
    df = pd.read_csv(input_file)
    logger.info(f"Date încărcate: {len(df)} companii")
    
    # Identifică companiile fără etichete
    unlabeled_mask = df['insurance_label'].isna() | (df['insurance_label'] == '')
    unlabeled_count = unlabeled_mask.sum()
    logger.info(f"Companii fără etichete: {unlabeled_count} din {len(df)} ({unlabeled_count/len(df)*100:.2f}%)")
    
    # Construiește mapările sectoare -> etichete
    logger.info("Construire mapări între sectoare și etichete...")
    mappings = build_sector_mappings(df, min_count=2)
    
    # Afișează statistici despre mapări
    logger.info(f"Sectoare mapate: {len(mappings['sector_to_labels'])}")
    logger.info(f"Categorii mapate: {len(mappings['category_to_labels'])}")
    logger.info(f"Nișe mapate: {len(mappings['niche_to_labels'])}")
    
    # Aplică clasificarea bazată pe sectoare
    logger.info("Aplicare clasificare bazată pe sectoare...")
    result_df, stats = apply_sector_classification(df, mappings, min_confidence=0.3, top_k=3)
    
    # Afișează statistici finale
    remaining_unlabeled = result_df['insurance_label'].isna() | (result_df['insurance_label'] == '')
    remaining_count = remaining_unlabeled.sum()
    
    logger.info(f"\nStatistici:")
    logger.info(f"Total companii procesate: {stats['total']}")
    logger.info(f"Companii nou clasificate: {stats['new']}")
    logger.info(f"Clasificări îmbunătățite: {stats['improved']}")
    logger.info(f"Companii rămase neclasificate: {remaining_count} din {len(df)} ({remaining_count/len(df)*100:.2f}%)")
    
    # Salvează rezultatele
    output_file = 'results/classified_companies_final.csv'
    result_df.to_csv(output_file, index=False)
    logger.info(f"Rezultate finale salvate în {output_file}")
    
    # Afișează câteva exemple
    if stats['new'] > 0:
        logger.info("\nExemple de companii nou clasificate cu abordarea bazată pe sectoare:")
        newly_labeled = result_df[~unlabeled_mask & remaining_unlabeled]
        for i, row in newly_labeled.head(3).iterrows():
            logger.info(f"\nCompanie: {row['description'][:100]}...")
            logger.info(f"Sector: {row.get('sector', 'N/A')}, Categorie: {row.get('category', 'N/A')}")
            logger.info(f"Etichete atribuite: {row['insurance_label']}")
            logger.info(f"Scoruri: {row['similarity_scores']}")
    
    if stats['improved'] > 0:
        logger.info("\nExemple de clasificări îmbunătățite:")
        improved_examples = result_df.iloc[range(100)].sample(3) if len(result_df) >= 100 else result_df.sample(min(3, len(result_df)))
        for i, row in improved_examples.iterrows():
            logger.info(f"\nCompanie: {row['description'][:100]}...")
            logger.info(f"Sector: {row.get('sector', 'N/A')}, Categorie: {row.get('category', 'N/A')}")
            logger.info(f"Etichete atribuite: {row['insurance_label']}")
            logger.info(f"Scoruri: {row['similarity_scores']}")

if __name__ == "__main__":
    main()
