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

# Download de benodigde NLTK data (voor stopwoorden) als deze nog niet aanwezig is.
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

print("=" * 80)
print("VKM SMART STUDY COACH - AI MODEL TRAINING")
print("=" * 80)

# STAP 1: DATA LADEN EN VOORBEREIDEN
print("\n[STAP 1] Data laden en voorbereiden...")
df = pd.read_csv('Opgeschoonde_VKM_dataset.csv')

print(f"Dataset geladen: {df.shape[0]} modules, {df.shape[1]} kolommen")
print(f"Kolommen: {', '.join(df.columns[:10])}...")

# Reset de index om zeker te zijn dat de rijen overeenkomen met de latere similarity matrix.
df.reset_index(drop=True, inplace=True)

# STAP 2: FEATURE ENGINEERING (KENMERKEN CREËREN)
# In deze stap maken we de data geschikt voor het model door nieuwe, betekenisvolle kolommen (features) te creëren.
print("\n[STAP 2] Feature Engineering...")

# We combineren alle relevante tekstkolommen tot één grote tekst per module.
# Dit geeft het model de volledige context om modules te kunnen vergelijken.
df['combined_text'] = (
    df['name'].fillna('') + ' ' + 
    df['shortdescription'].fillna('') + ' ' + 
    df['description'].fillna('') + ' ' + 
    df['content'].fillna('') + ' ' +
    df['module_tags'].fillna('')
).str.lower()

# Numerieke kolommen worden genormaliseerd. Dit betekent dat we ze allemaal op dezelfde schaal (0 tot 1) brengen.
# Dit voorkomt dat een kolom met grote getallen (zoals studycredit) meer invloed heeft dan een kolom met kleine getallen.
numeric_features = ['studycredit', 'interests_match_score', 'popularity_score', 
                   'estimated_difficulty', 'available_spots']

scaler = MinMaxScaler()
df_numeric = df[numeric_features].fillna(df[numeric_features].mean())
df_numeric_scaled = pd.DataFrame(
    scaler.fit_transform(df_numeric),
    columns=[f'{col}_scaled' for col in numeric_features]
)

for col in numeric_features:
    df[f'{col}_normalized'] = df_numeric_scaled[f'{col}_scaled'].values
print(f"'combined_text' kolom aangemaakt voor tekstanalyse.")
print(f"{len(numeric_features)} numerieke kolommen genormaliseerd (schaal 0-1).")

# STAP 3: TEKST OMZETTEN NAAR GETALLEN (TF-IDF VECTORIZATION)
# Een AI-model kan geen tekst lezen, dus we zetten de 'combined_text' om in een matrix van getallen.
print("\n[STAP 3] Tekst omzetten naar getallen met TF-IDF...")

# We gebruiken de Nederlandse stopwoordenlijst van NLTK om veelvoorkomende, niet-betekenisvolle woorden te negeren.
from nltk.corpus import stopwords
try:
    nl_stopwords = stopwords.words('dutch')
except LookupError:
    nltk.download('stopwords', quiet=True)
    nl_stopwords = stopwords.words('dutch')

# TF-IDF geeft een score aan elk woord, gebaseerd op hoe uniek en belangrijk het is voor een specifieke module.
tfidf = TfidfVectorizer(
    max_features=500,      # Beperk tot de 500 meest belangrijke woorden.
    ngram_range=(1, 2),    # Kijk naar losse woorden én combinaties van twee woorden (bv. "machine learning").
    min_df=2,              # Een woord moet in minimaal 2 modules voorkomen.
    max_df=0.8,            # Een woord mag in maximaal 80% van de modules voorkomen (filtert te algemene woorden).
    stop_words=nl_stopwords # Gebruik de Nederlandse stopwoordenlijst.
)

tfidf_matrix = tfidf.fit_transform(df['combined_text'])
print(f"TF-IDF matrix aangemaakt: {tfidf_matrix.shape}")
print(f"Grootte van het vocabulair: {len(tfidf.vocabulary_)} unieke termen")

# STAP 4: DIMENSIONALITEITSREDUCTIE MET SVD
# De TF-IDF matrix is erg groot. Met SVD reduceren we het aantal dimensies om ruis te verminderen en de berekeningen te versnellen.
print("\n[STAP 4] Dimensionaliteit reduceren met SVD...")

svd = TruncatedSVD(n_components=120, random_state=42)
tfidf_reduced = svd.fit_transform(tfidf_matrix)

print(f"Aantal dimensies gereduceerd van {tfidf_matrix.shape[1]} naar {tfidf_reduced.shape[1]}")
print(f"Verklaarde variantie door de nieuwe dimensies: {svd.explained_variance_ratio_.sum():.2%}")

# STAP 5: FEATURES COMBINEREN
# We voegen de gereduceerde tekst-features en de genormaliseerde numerieke features samen tot één finale feature matrix.
print("\n[STAP 5] Features combineren...")

# We geven de tekst-features een iets zwaarder gewicht omdat de inhoud het belangrijkst is voor de aanbeveling.
TEXT_WEIGHT = 0.6
NUMERIC_WEIGHT = 0.4

combined_features = np.hstack([
    tfidf_reduced * TEXT_WEIGHT,
    df_numeric_scaled.values * NUMERIC_WEIGHT
])

print(f"Finale feature matrix aangemaakt: {combined_features.shape}")
print(f"  - Tekst-features: {tfidf_reduced.shape[1]} (gewicht: {TEXT_WEIGHT})")
print(f"  - Numerieke features: {df_numeric_scaled.shape[1]} (gewicht: {NUMERIC_WEIGHT})")

# STAP 6: GELIJKENIS-MATRIX BEREKENEN (SIMILARITY MATRIX)
# We berekenen voor elke module hoe veel deze lijkt op elke andere module.
print("\n[STAP 6] Similarity matrix berekenen...")

# Cosine similarity berekent de hoek tussen twee vectoren; een kleine hoek (score dicht bij 1) betekent veel gelijkenis.
similarity_matrix = cosine_similarity(combined_features)
print(f"Similarity matrix aangemaakt: {similarity_matrix.shape}")
print(f"Gemiddelde gelijkenis: {similarity_matrix.mean():.3f}")

# STAP 7: MODULES CLUSTEREN
# We groeperen de modules in clusters op basis van hun gecombineerde features.
print("\n[STAP 7] Modules clusteren met KMeans...")

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(combined_features)

print(f"{len(df['cluster'].unique())} clusters geïdentificeerd.")
print("\nVerdeling van modules over de clusters:")
for cluster_id in sorted(df['cluster'].unique()):
    count = (df['cluster'] == cluster_id).sum()
    print(f"  Cluster {cluster_id}: {count} modules ({count/len(df)*100:.1f}%)")

# STAP 8: MODELVALIDATIE (SNELLE CONTROLE)
# Een simpele test om te zien of het model logische aanbevelingen geeft.
print("\n[STAP 8] Model validatie (steekproef)...")

test_idx = np.random.randint(0, len(df))
test_module = df.iloc[test_idx]

similarities = similarity_matrix[test_idx]
top_similar_indices = similarities.argsort()[-6:][::-1][1:]  # Top 5, exclusief de module zelf

print(f"\nTestmodule: '{test_module['name']}'")
print(f"Niveau: {test_module['level']}, Studiepunten: {test_module['studycredit']}")
print(f"\nTop 5 meest vergelijkbare modules volgens het model:")
for rank, idx in enumerate(top_similar_indices, 1):
    sim_module = df.iloc[idx]
    print(f"  {rank}. {sim_module['name']}")
    print(f"     Gelijkenis-score: {similarities[idx]:.3f} | Niveau: {sim_module['level']} | Studiepunten: {sim_module['studycredit']}")

# STAP 9: MODEL OPSLAAN (PERSISTENCE)
# We slaan alle getrainde componenten (het 'brein' van de AI) op in een .pkl-bestand.
print("\n[STAP 9] Model en andere componenten opslaan...")

# Zorg dat er een 'module_id' kolom is voor de herkenbaarheid.
if 'module_id' not in df.columns:
    df.insert(0, 'module_id', range(1, len(df) + 1))
    print("INFO: 'module_id' niet gevonden, fallback 'module_id' (1..N) toegevoegd.")

# Selecteer de kolommen die we in het uiteindelijke model willen bewaren.
desired_cols = ['module_id', 'name', 'level', 'studycredit',
                'interests_match_score', 'popularity_score',
                'estimated_difficulty', 'location', 'cluster']
available_cols = [c for c in desired_cols if c in df.columns]
missing = [c for c in desired_cols if c not in available_cols]
if missing:
    print(f"WAARSCHUWING: De volgende kolommen ontbreken en worden overgeslagen: {missing}")

# Bundel alle onderdelen van het model in een dictionary.
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

print("Model-componenten opgeslagen in 'vkm_recommender_model.pkl'")

# Sla de verwerkte dataset op, inclusief de nieuwe cluster-informatie.
df.to_csv('VKM_processed_with_clusters.csv', index=False)
print("Verwerkte dataset opgeslagen in 'VKM_processed_with_clusters.csv'")

# STAP 10: MODELSTATISTIEKEN EN INZICHTEN
# Een kijkje in wat het model heeft geleerd.
print("\n[STAP 10] Model statistieken en inzichten...")

print("\n Belangrijkste termen (Top 10 TF-IDF woorden):")
feature_names = tfidf.get_feature_names_out()
tfidf_scores = tfidf_matrix.sum(axis=0).A1
top_indices = tfidf_scores.argsort()[-10:][::-1]
for idx in top_indices:
    print(f"  - {feature_names[idx]}: {tfidf_scores[idx]:.2f}")

print("\n Clusterkarakteristieken:")
for cluster_id in sorted(df['cluster'].unique()):
    cluster_modules = df[df['cluster'] == cluster_id]
    print(f"\nCluster {cluster_id} ({len(cluster_modules)} modules):")
    print(f"  Gemiddelde moeilijkheid: {cluster_modules['estimated_difficulty'].mean():.2f}")
    print(f"  Gemiddelde populariteit: {cluster_modules['popularity_score'].mean():.2f}")
    print(f"  Meest voorkomend niveau: {cluster_modules['level'].mode()[0]}")
    print(f"  Voorbeeldmodules: {', '.join(cluster_modules['name'].head(3).tolist())}")

print("\n" + "=" * 80)
print(" AI MODEL TRAINING VOLTOOID")
print("=" * 80)
print("\nHet model is getraind en klaar voor gebruik in 'vkm_inference.py'.")
print("Gebruik het model met: pickle.load(open('vkm_recommender_model.pkl', 'rb'))")