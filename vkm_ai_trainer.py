import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import TruncatedSVD
import pickle
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data indien nodig
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

print("=" * 80)
print("VKM SMART STUDY COACH - AI MODEL TRAINING")
print("=" * 80)

# 1. DATA LOADING & PREPARATION
print("\n[STAP 1] Data laden en voorbereiden...")
df = pd.read_csv('Opgeschoonde_VKM_dataset.csv')

print(f"✓ Dataset geladen: {df.shape[0]} modules, {df.shape[1]} features")
print(f"✓ Kolommen: {', '.join(df.columns[:10])}...")

# Zorg dat alle rijen exact aligned zijn met similarity matrix
df.reset_index(drop=True, inplace=True)

# 2. FEATURE ENGINEERING
print("\n[STAP 2] Feature Engineering...")

# Tekstuele features combineren voor content-based filtering
df['combined_text'] = (
    df['name'].fillna('') + ' ' + 
    df['shortdescription'].fillna('') + ' ' + 
    df['description'].fillna('') + ' ' + 
    df['content'].fillna('') + ' ' +
    df['module_tags'].fillna('')
).str.lower()

# Numerieke features normaliseren
numeric_features = ['studycredit', 'interests_match_score', 'popularity_score', 
                   'estimated_difficulty', 'available_spots']

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

df_numeric = df[numeric_features].fillna(df[numeric_features].mean())
df_numeric_scaled = pd.DataFrame(
    scaler.fit_transform(df_numeric),
    columns=[f'{col}_scaled' for col in numeric_features]
)

for col in numeric_features:
    df[f'{col}_normalized'] = df_numeric_scaled[f'{col}_scaled'].values
print(f"✓ Combined text feature aangemaakt")
print(f"✓ {len(numeric_features)} numerieke features genormaliseerd")

# 3. TEXT VECTORIZATION (TF-IDF)
print("\n[STAP 3] Tekstuele data vectoriseren met TF-IDF...")

# TF-IDF voor Nederlandse tekst
# Gebruik de NLTK Nederlandse stopwoordenlijst en geef een list door aan scikit-learn
from nltk.corpus import stopwords
try:
    nl_stopwords = stopwords.words('dutch')
except LookupError:
    # Als de stopwords nog niet gedownload zijn, download ze en laad opnieuw
    nltk.download('stopwords', quiet=True)
    nl_stopwords = stopwords.words('dutch')

tfidf = TfidfVectorizer(
    max_features=500,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.8,
    stop_words=nl_stopwords
)

tfidf_matrix = tfidf.fit_transform(df['combined_text'])
print(f"✓ TF-IDF matrix aangemaakt: {tfidf_matrix.shape}")
print(f"✓ Vocabulaire grootte: {len(tfidf.vocabulary_)} unieke termen")

# 4. DIMENSIONALITY REDUCTION
print("\n[STAP 4] Dimensionaliteitsreductie met SVD...")

svd = TruncatedSVD(n_components=120, random_state=42)
tfidf_reduced = svd.fit_transform(tfidf_matrix)

print(f"✓ TF-IDF gereduceerd van {tfidf_matrix.shape[1]} naar {tfidf_reduced.shape[1]} dimensies")
print(f"✓ Verklaarde variantie: {svd.explained_variance_ratio_.sum():.2%}")

# 5. COMBINE FEATURES
print("\n[STAP 5] Features combineren...")

# Gewichten voor verschillende feature types
TEXT_WEIGHT = 0.6
NUMERIC_WEIGHT = 0.4

# Combineer text embeddings met numerieke features
combined_features = np.hstack([
    tfidf_reduced * TEXT_WEIGHT,
    df_numeric_scaled.values * NUMERIC_WEIGHT
])

print(f"✓ Feature matrix: {combined_features.shape}")
print(f"  - Text features: {tfidf_reduced.shape[1]} (gewicht: {TEXT_WEIGHT})")
print(f"  - Numeric features: {df_numeric_scaled.shape[1]} (gewicht: {NUMERIC_WEIGHT})")

# 6. SIMILARITY MATRIX COMPUTATION
print("\n[STAP 6] Similarity matrix berekenen...")

similarity_matrix = cosine_similarity(combined_features)
print(f"✓ Similarity matrix: {similarity_matrix.shape}")
print(f"✓ Gemiddelde similarity: {similarity_matrix.mean():.3f}")
print(f"✓ Min similarity: {similarity_matrix.min():.3f}")
print(f"✓ Max similarity: {similarity_matrix.max():.3f}")

# 7. MODULE CLUSTERING ANALYSIS
print("\n[STAP 7] Module clusters identificeren...")

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(combined_features)

print(f"✓ {len(df['cluster'].unique())} clusters geïdentificeerd")
print("\nCluster distributie:")
for cluster_id in sorted(df['cluster'].unique()):
    count = (df['cluster'] == cluster_id).sum()
    print(f"  Cluster {cluster_id}: {count} modules ({count/len(df)*100:.1f}%)")

# 8. MODEL VALIDATION
print("\n[STAP 8] Model validatie...")

# Test: vind vergelijkbare modules voor een random sample
test_idx = np.random.randint(0, len(df))
test_module = df.iloc[test_idx]

similarities = similarity_matrix[test_idx]
top_similar_indices = similarities.argsort()[-6:][::-1][1:]  # Exclude zelf

print(f"\nTest module: '{test_module['name']}'")
print(f"Level: {test_module['level']}, Credits: {test_module['studycredit']}")
print(f"\nTop 5 vergelijkbare modules:")
for rank, idx in enumerate(top_similar_indices, 1):
    sim_module = df.iloc[idx]
    print(f"  {rank}. {sim_module['name']}")
    print(f"     Similarity: {similarities[idx]:.3f} | Level: {sim_module['level']} | Credits: {sim_module['studycredit']}")

# 9. MODEL PERSISTENCE
print("\n[STAP 9] Model en artifacts opslaan...")

# Zorg dat er een 'module_id' kolom is; als deze niet bestaat, maak een fallback aan
if 'module_id' not in df.columns:
    # Eenvoudige, stabiele fallback id (1-based index). Dit voorkomt KeyError bij opslaan.
    df.insert(0, 'module_id', range(1, len(df) + 1))
    print("'module_id' niet gevonden in dataset — fallback 'module_id' toegevoegd (1..N)")

# Kies de kolommen die we willen bewaren voor inference; alleen bestaande kolommen worden meegenomen
desired_cols = ['module_id', 'name', 'level', 'studycredit',
                'interests_match_score', 'popularity_score',
                'estimated_difficulty', 'location', 'cluster']
available_cols = [c for c in desired_cols if c in df.columns]
missing = [c for c in desired_cols if c not in available_cols]
if missing:
    print(f" Volgende verwachte kolommen ontbreken en worden overgeslagen: {missing}")

# Save alle belangrijke componenten
model_artifacts = {
    'tfidf_vectorizer': tfidf,
    'svd_model': svd,
    'scaler': scaler,
    'similarity_matrix': similarity_matrix,
    'kmeans_model': kmeans,
    'feature_columns': numeric_features,
    'text_weight': TEXT_WEIGHT,
    'numeric_weight': NUMERIC_WEIGHT,
    'dataframe': df[available_cols].copy()
}

with open('vkm_recommender_model.pkl', 'wb') as f:
    pickle.dump(model_artifacts, f)

print("✓ Model artifacts opgeslagen in 'vkm_recommender_model.pkl'")

# Save processed dataset
df.to_csv('VKM_processed_with_clusters.csv', index=False)
print("✓ Processed dataset opgeslagen in 'VKM_processed_with_clusters.csv'")

# 10. MODEL STATISTICS & INSIGHTS
print("\n[STAP 10] Model statistieken en inzichten...")

print("\n FEATURE IMPORTANCE (Top 10 TF-IDF termen):")
feature_names = tfidf.get_feature_names_out()
tfidf_scores = tfidf_matrix.sum(axis=0).A1
top_indices = tfidf_scores.argsort()[-10:][::-1]
for idx in top_indices:
    print(f"  - {feature_names[idx]}: {tfidf_scores[idx]:.2f}")

print("\n CLUSTER KARAKTERISTIEKEN:")
for cluster_id in sorted(df['cluster'].unique()):
    cluster_modules = df[df['cluster'] == cluster_id]
    print(f"\nCluster {cluster_id} ({len(cluster_modules)} modules):")
    print(f"  Gemiddelde moeilijkheid: {cluster_modules['estimated_difficulty'].mean():.2f}")
    print(f"  Gemiddelde populariteit: {cluster_modules['popularity_score'].mean():.2f}")
    print(f"  Meest voorkomend level: {cluster_modules['level'].mode()[0]}")
    print(f"  Voorbeeldmodules: {', '.join(cluster_modules['name'].head(3).tolist())}")

print("\n" + "=" * 80)
print(" AI MODEL TRAINING VOLTOOID")
print("=" * 80)
print("\nModel ready voor inference!")
print("Gebruik het model met: pickle.load(open('vkm_recommender_model.pkl', 'rb'))")