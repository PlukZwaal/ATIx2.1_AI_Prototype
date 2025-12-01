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
        # Nederlandse stopwoorden (klein lijstje is genoeg)
        dutch_stopwords = {'de', 'het', 'een', 'en', 'van', 'in', 'op', 'met', 'voor', 'te', 'is', 'ik', 'je', 'mijn',
                           'aan', 'uit', 'over', 'door', 'bij', 'als', 'wat', 'wie', 'hoe', 'niet', 'wel', 'dan',
                           'of', 'maar', 'toch', 'ook', 'nog', 'al', 'alleen', 'zo', 'ze', 'zij', 'hij'}

        # Student woorden (zonder stopwoorden en leestekens)
        student_words = {w.lower() for w in student_text.replace(',', ' ').replace('.', ' ').split()
                        if len(w) > 2 and w.lower() not in dutch_stopwords}

        # Module woorden (naam + korte beschrijving + tags)
        module_text = f"{row['name']} {row.get('shortdescription','')} {row.get('module_tags','')}".lower()
        module_words = {w for w in module_text.replace(',', ' ').replace('.', ' ').split()
                       if len(w) > 2 and w not in dutch_stopwords}

        # Overlappende inhoudelijke woorden
        overlap = student_words.intersection(module_words)

        if len(overlap) >= 2:
            top_words = sorted(overlap)[:5]
            return f"Sterke match op: {', '.join(top_words)}"

        if len(overlap) == 1:
            return f"Match op woord: {list(overlap)[0]}"

        # Fallback: domein-herkenning (zoals je al had)
        text = student_text.lower()

        if any(k in text for k in ['ai', 'machine learning', 'python', 'data', 'programmeren', 'deep learning']):
            return "Perfecte match met je interesse in AI & data!"

        if any(k in text for k in ['zorg', 'psychologie', 'coaching', 'mensen helpen', 'welzijn', 'patiënt']):
            return "Ideaal voor jouw wens om mensen te helpen"

        if any(k in text for k in ['ondernemen', 'business', 'marketing', 'bedrijf', 'sales', 'startup']):
            return "Super voor je ondernemersambities!"

        if row['popularity_norm'] > 0.8:
            return "Zeer populaire module bij medestudenten"

        return "Goede algemene match met jouw interesses"

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