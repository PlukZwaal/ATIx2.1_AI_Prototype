import pandas as pd
import numpy as np
import pickle

print("=" * 80)
print("VKM SMART STUDY COACH - AI INFERENCE ENGINE")
print("=" * 80)

# Load het getrainde model
print("\n[1] Model laden...")
with open('vkm_recommender_model.pkl', 'rb') as f:
    model = pickle.load(f)

print("Model geladen")
print(f"  - {len(model['dataframe'])} modules beschikbaar")
print(f"  - {model['similarity_matrix'].shape[0]}x{model['similarity_matrix'].shape[1]} similarity matrix")

# Load de volledige dataset voor details
df_full = pd.read_csv('VKM_processed_with_clusters.csv')


class VKMRecommender:
    """AI-gestuurd VKM Aanbevelingssysteem"""
    
    def __init__(self, model_artefacten, volledig_dataframe):
        self.similarity_matrix = model_artefacten['similarity_matrix']
        self.df = volledig_dataframe
        self.df_mini = model_artefacten['dataframe']
        
    def get_recommendations(self, module_index, top_n=5, filters=None):
        """
        Geeft aanbevelingen op basis van de index van een module.
        
        Parameters:
        -----------
        module_index : int
            De index van de module die als referentie dient.
        top_n : int
            Het aantal aanbevelingen dat moet worden teruggegeven.
        filters : dict
            Optionele filters om de resultaten te verfijnen (bijv. {'level': 'B', 'min_spots': 5}).
        """
        similarities = self.similarity_matrix[module_index]
        
        # Filter de modules indien er filters zijn opgegeven
        valid_indices = np.arange(len(self.df))
        
        if filters:
            mask = np.ones(len(self.df), dtype=bool)
            
            if 'level' in filters:
                mask &= (self.df['level'] == filters['level']).values
            if 'min_spots' in filters:
                mask &= (self.df['available_spots'] >= filters['min_spots']).values
            if 'max_difficulty' in filters:
                mask &= (self.df['estimated_difficulty'] <= filters['max_difficulty']).values
            if 'min_popularity' in filters:
                mask &= (self.df['popularity_score'] >= filters['min_popularity']).values
                
            valid_indices = np.where(mask)[0]
        
        # Haal de meest vergelijkbare modules op (sluit de referentiemodule zelf uit)
        valid_similarities = [(idx, similarities[idx]) for idx in valid_indices if idx != module_index]
        valid_similarities.sort(key=lambda x: x[1], reverse=True)
        
        top_indices = [idx for idx, _ in valid_similarities[:top_n]]
        
        return self.df.iloc[top_indices]
    
    def get_recommendations_by_profile(self, student_profile, top_n=5):
        """
        Geeft aanbevelingen op basis van een studentenprofiel.

        Parameters:
        -----------
        student_profile : dict
            Een dictionary met studentkenmerken, bijv:
            {'interests': ['data', 'ai'], 'preferred_difficulty': 3, 'level': 'B'}
        """
        eps = 1e-8  # Een kleine waarde om deling door nul te voorkomen
        n = len(self.df)
        scores = np.zeros(n)

        # Stap 1: Match op basis van interesses (trefwoorden)
        interest_counts = np.zeros(n, dtype=float)
        if 'interests' in student_profile and student_profile['interests']:
            interests = [s.lower() for s in student_profile['interests']]
            for idx, combined in enumerate(self.df['combined_text'].fillna('').astype(str)):
                cnt = sum(1 for it in interests if it in combined.lower())
                interest_counts[idx] = cnt
        # Normaliseer de interesse-scores naar een schaal van 0-1
        ic_min, ic_max = interest_counts.min(), interest_counts.max()
        if ic_max - ic_min > 0:
            ic_norm = (interest_counts - ic_min) / (ic_max - ic_min + eps)
        else:
            ic_norm = interest_counts  # Allemaal nullen als er geen verschil is
        scores += ic_norm * 2.0  # Geef extra gewicht aan trefwoord-matches (aanpasbaar)

        # Stap 2: Match op basis van gewenste moeilijkheidsgraad
        if 'preferred_difficulty' in student_profile:
            pref_diff = student_profile['preferred_difficulty']
            # Bereken de score: 1 min de absolute afwijking van de voorkeur
            diff_scores = 1 - np.abs(self.df['estimated_difficulty'].fillna(pref_diff).values - pref_diff) / 5
            diff_scores = np.clip(diff_scores, 0, 1) # Zorg dat de score tussen 0 en 1 blijft
            scores += diff_scores * 1.0 # Gewicht voor moeilijkheid (aanpasbaar)

        # Stap 3: Match op basis van niveau
        if 'level' in student_profile:
            level_match = (self.df['level'] == student_profile['level']).astype(int).values
            scores += level_match * 1.5 # Geef extra gewicht als het level overeenkomt (aanpasbaar)

        # Stap 4: Gebruik de opgeslagen populariteits- en interesse-scores
        # Normaliseer deze scores ook naar een schaal van 0-1 voor een eerlijke vergelijking
        pop = self.df['popularity_score'].fillna(0).astype(float).values
        pop_min, pop_max = pop.min(), pop.max()
        pop_norm = (pop - pop_min) / (pop_max - pop_min + eps)

        ims = self.df['interests_match_score'].fillna(0).astype(float).values
        ims_min, ims_max = ims.min(), ims.max()
        ims_norm = (ims - ims_min) / (ims_max - ims_min + eps)

        # Voeg de genormaliseerde scores toe aan de totaalscore met hun eigen gewichten
        scores += ims_norm * 0.5
        scores += pop_norm * 0.3

        # Haal de top N modules met de hoogste totaalscores op
        top_indices = scores.argsort()[-top_n:][::-1]

        results = self.df.iloc[top_indices].copy()
        results['match_score'] = scores[top_indices]

        return results
    

    def get_cluster_recommendations(self, cluster_id, top_n=5):
        """Geeft de topmodules uit een specifiek cluster, gesorteerd op een gewogen score."""
        cluster_modules = self.df[self.df['cluster'] == cluster_id].copy()
        
        eps = 1e-8 # Kleine waarde om deling door nul te voorkomen
        # Gebruik de min/max van de *gehele* dataset voor een eerlijke normalisatie
        pop_min = self.df['popularity_score'].min()
        pop_max = self.df['popularity_score'].max()
        int_min = self.df['interests_match_score'].min()
        int_max = self.df['interests_match_score'].max()
        
        # Normaliseer populariteit en interesse-match score naar een schaal van 0-1
        cluster_modules['pop_norm'] = (cluster_modules['popularity_score'] - pop_min) / (pop_max - pop_min + eps)
        cluster_modules['int_norm'] = (cluster_modules['interests_match_score'] - int_min) / (int_max - int_min + eps)
        
        # Pas gewichten toe om een eindscore te berekenen (deze zijn aanpasbaar)
        cluster_modules['score'] = cluster_modules['pop_norm'] * 0.5 + cluster_modules['int_norm'] * 0.5
        
        # Geef de N modules met de hoogste score terug
        return cluster_modules.nlargest(top_n, 'score')


# Initialiseer de recommender
print("\n[2] Recommender initialiseren...")
recommender = VKMRecommender(model, df_full)
print("Recommender is klaar voor gebruik")

# DEMO 1: Content-based aanbevelingen (op basis van vergelijkbare modules)
print("\n" + "=" * 80)
print("DEMO 1: CONTENT-BASED AANBEVELINGEN")
print("=" * 80)

# Kies een willekeurige module als uitgangspunt
base_idx = 10
base_module = df_full.iloc[base_idx]

print(f"\n Gekozen basismodule:")
print(f"   Naam: {base_module['name']}")
print(f"   Level: {base_module['level']} | Studiepunten: {base_module['studycredit']}")
print(f"   Moeilijkheid: {base_module['estimated_difficulty']}/5")
print(f"   Populariteit: {base_module['popularity_score']:.2f}")

print(f"\n Top 5 meest vergelijkbare modules:")
recommendations = recommender.get_recommendations(base_idx, top_n=5)

for i, (idx, module) in enumerate(recommendations.iterrows(), 1):
    similarity = model['similarity_matrix'][base_idx][idx]
    print(f"\n{i}. {module['name']}")
    print(f"   Gelijkenis-score: {similarity:.3f}")
    print(f"   Level: {module['level']} | Studiepunten: {module['studycredit']}")
    print(f"   Moeilijkheid: {module['estimated_difficulty']}/5 | Populariteit: {module['popularity_score']:.2f}")

# DEMO 2: Profiel-gebaseerde aanbevelingen
print("\n" + "=" * 80)
print("DEMO 2: PROFIEL-GEBASEERDE AANBEVELINGEN")
print("=" * 80)

student_profile = {
    'interests': ['data', 'analyse', 'machine learning', 'AI'],
    'preferred_difficulty': 3,
    'level': 'NLQF5'
}

print(f"\n Studentenprofiel:")
print(f"   Interesses: {', '.join(student_profile['interests'])}")
print(f"   Gewenste moeilijkheid: {student_profile['preferred_difficulty']}/5")
print(f"   Niveau: {student_profile['level']}")

print(f"\n Top 5 aanbevelingen voor dit profiel:")
profile_recs = recommender.get_recommendations_by_profile(student_profile, top_n=5)

for i, (idx, module) in enumerate(profile_recs.iterrows(), 1):
    print(f"\n{i}. {module['name']}")
    print(f"   Match-score: {module['match_score']:.2f}")
    print(f"   Level: {module['level']} | Studiepunten: {module['studycredit']}")
    print(f"   Moeilijkheid: {module['estimated_difficulty']}/5 | Populariteit: {module['popularity_score']:.2f}")
    print(f"   Locatie: {module['location']}")

# DEMO 3: Cluster-gebaseerde aanbevelingen
print("\n" + "=" * 80)
print("DEMO 3: CLUSTER-GEBASEERDE AANBEVELINGEN")
print("=" * 80)

cluster_id = 2
print(f"\n Populairste modules uit Cluster {cluster_id}:")

cluster_recs = recommender.get_cluster_recommendations(cluster_id, top_n=5)

for i, (idx, module) in enumerate(cluster_recs.iterrows(), 1):
    print(f"\n{i}. {module['name']}")
    print(f"   Gewogen score: {module['score']:.2f}")
    print(f"   Level: {module['level']} | Studiepunten: {module['studycredit']}")
    print(f"   Moeilijkheid: {module['estimated_difficulty']}/5 | Populariteit: {module['popularity_score']:.2f}")

print("\n" + "=" * 80)
print(" INFERENCE VOLTOOID")
print("=" * 80)
print("\nHet model kan nu gebruikt worden voor real-time aanbevelingen!")