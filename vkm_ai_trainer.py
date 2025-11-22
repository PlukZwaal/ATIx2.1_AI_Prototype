import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import pickle
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

print("=" * 80)
print("VKM SMART STUDY COACH - STUDENT-TO-MODULE AI MODEL")
print("=" * 80)

# STAP 1: DATA LADEN
print("\n[STAP 1] Data laden...")
df = pd.read_csv('Opgeschoonde_VKM_dataset.csv')
print(f"Dataset geladen: {df.shape[0]} modules")
df.reset_index(drop=True, inplace=True)

# STAP 2: FEATURE ENGINEERING
print("\n[STAP 2] Feature Engineering...")

# Combineer alle tekstkolommen
df['combined_text'] = (
    df['name'].fillna('') + ' ' + 
    df['shortdescription'].fillna('') + ' ' + 
    df['description'].fillna('') + ' ' + 
    df['content'].fillna('') + ' ' +
    df['module_tags'].fillna('')
).str.lower()

# Normaliseer numerieke features
numeric_features = ['studycredit', 'interests_match_score', 'popularity_score', 
                   'estimated_difficulty', 'available_spots']
scaler = MinMaxScaler()
df_numeric = df[numeric_features].fillna(df[numeric_features].mean())
df_numeric_scaled = scaler.fit_transform(df_numeric)

for i, col in enumerate(numeric_features):
    df[f'{col}_normalized'] = df_numeric_scaled[:, i]

print(f"Combined text kolom aangemaakt")
print(f"{len(numeric_features)} numerieke kolommen genormaliseerd")

# STAP 3: TF-IDF VECTORIZATION (dit is de KERN van de recommender!)
print("\n[STAP 3] TF-IDF Vectorization - DRIE VARIANTEN TESTEN...")

from nltk.corpus import stopwords
nl_stopwords = stopwords.words('dutch')

# VARIANT 1: Baseline (met bigrams, 3000 features, met stopwoorden)
print("\n VARIANT 1: Baseline (bigrams, 3000 features, met stopwoorden)")
tfidf_v1 = TfidfVectorizer(
    max_features=3000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.8,
    stop_words=nl_stopwords
)
tfidf_matrix_v1 = tfidf_v1.fit_transform(df['combined_text'])
print(f"  TF-IDF matrix: {tfidf_matrix_v1.shape}")
print(f"  Vocabulair: {len(tfidf_v1.vocabulary_)} termen")
print(f"  Sparsity: {(1.0 - tfidf_matrix_v1.nnz / (tfidf_matrix_v1.shape[0] * tfidf_matrix_v1.shape[1])):.2%}")

# VARIANT 2: Alleen unigrams, 6000 features
print("\n VARIANT 2: Unigrams only, 6000 features, met stopwoorden")
tfidf_v2 = TfidfVectorizer(
    max_features=6000,
    ngram_range=(1, 1),  # Alleen losse woorden
    min_df=2,
    max_df=0.8,
    stop_words=nl_stopwords
)
tfidf_matrix_v2 = tfidf_v2.fit_transform(df['combined_text'])
print(f"  TF-IDF matrix: {tfidf_matrix_v2.shape}")
print(f"  Vocabulair: {len(tfidf_v2.vocabulary_)} termen")
print(f"  Sparsity: {(1.0 - tfidf_matrix_v2.nnz / (tfidf_matrix_v2.shape[0] * tfidf_matrix_v2.shape[1])):.2%}")

# VARIANT 3: Bigrams, 3000 features, ZONDER stopwoorden
print("\n VARIANT 3: Bigrams, 3000 features, ZONDER stopwoorden")
tfidf_v3 = TfidfVectorizer(
    max_features=3000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.8,
    stop_words=None  # Geen stopwoorden
)
tfidf_matrix_v3 = tfidf_v3.fit_transform(df['combined_text'])
print(f"  TF-IDF matrix: {tfidf_matrix_v3.shape}")
print(f"  Vocabulair: {len(tfidf_v3.vocabulary_)} termen")
print(f"  Sparsity: {(1.0 - tfidf_matrix_v3.nnz / (tfidf_matrix_v3.shape[0] * tfidf_matrix_v3.shape[1])):.2%}")

# STAP 4: VALIDATIE - TEST MET EEN VOORBEELD STUDENTPROFIEL
print("\n[STAP 4] Validatie - Test met studentprofiel...")

test_student_text = "Ik ben geïnteresseerd in data analyse machine learning AI kunstmatige intelligentie programmeren python"

print(f"\nTest studentprofiel: '{test_student_text}'")

for variant_num, (tfidf_model, tfidf_matrix) in enumerate([
    (tfidf_v1, tfidf_matrix_v1),
    (tfidf_v2, tfidf_matrix_v2),
    (tfidf_v3, tfidf_matrix_v3)
], 1):
    print(f"\n--- VARIANT {variant_num} RESULTATEN ---")
    
    # Vectoriseer het studentprofiel met DEZELFDE vectorizer
    student_vector = tfidf_model.transform([test_student_text])
    
    # Bereken cosine similarity tussen student en alle modules
    similarities = cosine_similarity(student_vector, tfidf_matrix)[0]
    
    # Haal top 3 modules op
    top_indices = similarities.argsort()[-3:][::-1]
    
    print(f"Top 3 aanbevelingen:")
    for rank, idx in enumerate(top_indices, 1):
        print(f"  {rank}. {df.iloc[idx]['name']}")
        print(f"     Similarity: {similarities[idx]:.3f} | Level: {df.iloc[idx]['level']} | Credits: {df.iloc[idx]['studycredit']}")

# STAP 5: DIMENSIONALITEITSREDUCTIE MET SVD
print("\n[STAP 5] Dimensionaliteitsreductie met SVD...")

from sklearn.decomposition import TruncatedSVD

# Kies Variant 1 als beste (bigrams werken goed voor Nederlandse tekst)
chosen_tfidf = tfidf_v1
chosen_matrix = tfidf_matrix_v1

print("\nGekozen: VARIANT 1 (bigrams, 3000 features, met stopwoorden)")
print("Reden: Goede balans tussen detail (bigrams) en efficiency (3000 features)")

# Pas SVD toe om dimensies te reduceren en ruis te verminderen
# We reduceren van 3000 naar 200 dimensies
svd = TruncatedSVD(n_components=200, random_state=42)
tfidf_reduced = svd.fit_transform(chosen_matrix)

print(f"\nSVD toegepast:")
print(f"  Originele dimensies: {chosen_matrix.shape[1]}")
print(f"  Gereduceerde dimensies: {tfidf_reduced.shape[1]}")
print(f"  Verklaarde variantie: {svd.explained_variance_ratio_.sum():.2%}")
print(f"  Reductie: {(1 - tfidf_reduced.shape[1]/chosen_matrix.shape[1])*100:.1f}%")

# Test: vergelijk aanbevelingen met en zonder SVD
print("\n Vergelijking: Origineel vs SVD-gereduceerd")
test_text = "Ik ben geïnteresseerd in data analyse machine learning"
student_vec_original = chosen_tfidf.transform([test_text])
student_vec_reduced = svd.transform(student_vec_original)

# Similarities zonder SVD
sim_original = cosine_similarity(student_vec_original, chosen_matrix)[0]
top_original = sim_original.argsort()[-3:][::-1]

# Similarities met SVD
sim_reduced = cosine_similarity(student_vec_reduced, tfidf_reduced)[0]
top_reduced = sim_reduced.argsort()[-3:][::-1]

print("\nTop 3 zonder SVD:")
for i, idx in enumerate(top_original, 1):
    print(f"  {i}. {df.iloc[idx]['name']} (score: {sim_original[idx]:.3f})")

print("\nTop 3 met SVD:")
for i, idx in enumerate(top_reduced, 1):
    print(f"  {i}. {df.iloc[idx]['name']} (score: {sim_reduced[idx]:.3f})")

# STAP 6: MODEL OPSLAAN
print("\n[STAP 6] Model opslaan...")

# Voeg module_id toe indien nodig
if 'module_id' not in df.columns:
    df.insert(0, 'module_id', range(1, len(df) + 1))

# Bewaar alleen essentiële kolommen
essential_cols = ['module_id', 'name', 'level', 'studycredit', 'shortdescription',
                  'interests_match_score', 'popularity_score', 'estimated_difficulty', 
                  'location', 'combined_text']
available_cols = [c for c in essential_cols if c in df.columns]

# Bundel model artifacts
model_artifacts = {
    'tfidf_vectorizer': chosen_tfidf,
    'tfidf_matrix': chosen_matrix,
    'scaler': scaler,
    'feature_columns': numeric_features,
    'dataframe': df[available_cols].copy(),
    'model_version': 'student-to-module-v1',
    'tuning_notes': {
        'variant_1': 'bigrams, 3000 features, stopwoorden - GEKOZEN',
        'variant_2': 'unigrams, 6000 features - te generiek',
        'variant_3': 'bigrams, geen stopwoorden - te veel ruis'
    }
}

with open('vkm_student_recommender_model.pkl', 'wb') as f:
    pickle.dump(model_artifacts, f)

print("\nModel opgeslagen: 'vkm_student_recommender_model.pkl'")

# STAP 6: MODEL STATISTIEKEN
print("\n[STAP 6] Model statistieken...")

print("\nTop 10 belangrijkste termen in vocabulair:")
feature_names = chosen_tfidf.get_feature_names_out()
tfidf_scores = chosen_matrix.sum(axis=0).A1
top_indices = tfidf_scores.argsort()[-10:][::-1]
for idx in top_indices:
    print(f"  - {feature_names[idx]}: {tfidf_scores[idx]:.2f}")

print("\n" + "=" * 80)
print("STUDENT-TO-MODULE MODEL TRAINING VOLTOOID")
print("=" * 80)
print("\nDit model kan nu studenten matchen met modules op basis van hun interesses!")