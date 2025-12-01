# train_final_model.py → maakt een PERFECT picklebaar model
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import pickle
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)
nl_stopwords = stopwords.words('dutch')

print("=" * 80)
print("TRAINING: FINALE HYBRIDE MODEL (70% TF-IDF + 15% interests + 15% popularity)")
print("=" * 80)

# 1. Data laden
df = pd.read_csv('Opgeschoonde_VKM_dataset.csv')
df.reset_index(drop=True, inplace=True)

# 2. Tekst combineren
df['combined_text'] = (
    df['name'].fillna('') + ' ' +
    df['shortdescription'].fillna('') + ' ' +
    df['description'].fillna('') + ' ' +
    df['content'].fillna('') + ' ' +
    df['module_tags'].fillna('')
).str.lower()

# 3. TF-IDF (unigrams)
tfidf = TfidfVectorizer(
    max_features=3000,
    ngram_range=(1,1),
    min_df=2,
    max_df=0.8,
    stop_words=nl_stopwords
)
tfidf_matrix = tfidf.fit_transform(df['combined_text'])

# 4. Normaliseer de twee kolommen die we nodig hebben
scaler = MinMaxScaler()
df[['interests_norm', 'popularity_norm']] = scaler.fit_transform(
    df[['interests_match_score', 'popularity_score']].fillna(0)
)

# 5. Voeg module_id toe
df['module_id'] = range(1, len(df)+1)

# 6. OPSLAAN – GEEN FUNCTIES MEER! (pickle-veilig)
model_artifacts = {
    'tfidf_vectorizer': tfidf,
    'tfidf_matrix': tfidf_matrix,
    'dataframe': df.copy(),
    'model_version': 'VKM_HYBRID_70_15_15_FINAL',
    'description': '70% TF-IDF + 15% interests + 15% popularity – class-ready'
}

with open('vkm_student_recommender_model.pkl', 'wb') as f:
    pickle.dump(model_artifacts, f)