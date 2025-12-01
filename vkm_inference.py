import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

print("=" * 90)
print("VKM SMART STUDY COACH - LIVE AANBEVELINGEN")
print("=" * 90)

# Model laden
with open('vkm_student_recommender_model.pkl', 'rb') as f:
    model = pickle.load(f)

print(f"Model: {model['model_version']}")
print(f"Modules beschikbaar: {len(model['dataframe'])}\n")

class StudentToModuleRecommender:
    def __init__(self, artifacts):
        self.vectorizer = artifacts['tfidf_vectorizer']
        self.matrix = artifacts['tfidf_matrix']
        self.df = artifacts['dataframe']

    def recommend(self, student_text, top_n=5, filters=None, explain=True):
        # TF-IDF similarity
        vec = self.vectorizer.transform([student_text.lower()])
        sims = cosine_similarity(vec, self.matrix)[0]

        # Hybride score: 70% TF-IDF + 15% interests + 15% popularity
        final_scores = (
            0.70 * sims +
            0.15 * self.df['interests_norm'].values +
            0.15 * self.df['popularity_norm'].values
        )

        # Filters toepassen
        mask = np.ones(len(self.df), dtype=bool)
        if filters:
            if 'level' in filters:
                mask &= (self.df['level'] == filters['level'])
            if 'min_credits' in filters:
                mask &= (self.df['studycredit'] >= filters['min_credits'])
            if 'location' in filters:
                mask &= (self.df['location'] == filters['location'])

        candidates = np.where(mask)[0]
        top_idx = candidates[np.argsort(final_scores[candidates])[-top_n:][::-1]]

        results = self.df.iloc[top_idx].copy().reset_index(drop=True)
        results['match_score'] = final_scores[top_idx].round(4)
        results['content_similarity'] = sims[top_idx].round(4)

        if explain:
            results['explanation'] = results.apply(lambda row: self._explain(row, student_text), axis=1)

        return results[['name', 'level', 'studycredit', 'location', 'match_score', 'explanation']]

    def _explain(self, row, student_text):
        text = student_text.lower()
        module = f"{row['name']} {row.get('shortdescription', '')}".lower()

        # Correcte syntax: == in plaats van =
        any_ai = any(k in text for k in ['ai', 'machine learning', 'python', 'data', 'programmeren'])
        any_zorg = any(k in text for k in ['zorg', 'psycholog', 'coach', 'mensen helpen', 'welzijn'])
        any_business = any(k in text for k in ['business', 'ondernemen', 'marketing', 'finance', 'bedrijf'])

        if any_ai and any(k in module for k in ['ai', 'data', 'python', 'machine learning', 'intelligentie']):
            return "Perfecte match met je interesse in technologie & data!"
        if any_zorg and any(k in module for k in ['zorg', 'psycholog', 'coaching', 'welzijn']):
            return "Ideaal voor jouw wens om mensen te helpen"
        if any_business and any(k in module for k in ['business', 'ondernem', 'marketing', 'finance']):
            return "Super voor je ondernemersambities!"
        if row['popularity_norm'] > 0.8:
            return "Zeer populaire module onder studenten"
        return "Goede algemene match met jouw profiel"

# Start recommender
recommender = StudentToModuleRecommender(model)
print("Recommender actief!\n")

# DEMO'S
print("1. AI & DATA SCIENCE STUDENT")
print(recommender.recommend(
    "Ik wil alles leren over AI, machine learning, Python en data science",
    top_n=6
))

print("\n" + "="*90)

print("2. PSYCHOLOGIE & ZORG STUDENT")
print(recommender.recommend(
    "Ik hou van psychologie, coaching en mensen helpen in de zorg",
    top_n=6
))

print("\n" + "="*90)

print("3. ONDERNEMEN & MARKETING (alleen NLQF6)")
print(recommender.recommend(
    "Ik wil een eigen bedrijf starten, marketing en sales",
    top_n=5,
    filters={'level': 'NLQF6'}
))