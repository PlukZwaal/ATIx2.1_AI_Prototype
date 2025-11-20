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

print("✓ Model geladen")
print(f"  - {len(model['dataframe'])} modules beschikbaar")
print(f"  - {model['similarity_matrix'].shape[0]}x{model['similarity_matrix'].shape[1]} similarity matrix")

# Load de volledige dataset voor details
df_full = pd.read_csv('VKM_processed_with_clusters.csv')


class VKMRecommender:
    """AI-powered VKM Recommender System"""
    
    def __init__(self, model_artifacts, full_dataframe):
        self.similarity_matrix = model_artifacts['similarity_matrix']
        self.df = full_dataframe
        self.df_mini = model_artifacts['dataframe']
        
    def get_recommendations(self, module_index, top_n=5, filters=None):
        """
        Krijg aanbevelingen op basis van een module index
        
        Parameters:
        -----------
        module_index : int
            Index van de referentie module
        top_n : int
            Aantal aanbevelingen
        filters : dict
            Optionele filters (bijv. {'level': 'B', 'min_spots': 5})
        """
        similarities = self.similarity_matrix[module_index]
        
        # Filter modules indien gewenst
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
        
        # Get top similar modules (exclude de module zelf)
        valid_similarities = [(idx, similarities[idx]) for idx in valid_indices if idx != module_index]
        valid_similarities.sort(key=lambda x: x[1], reverse=True)
        
        top_indices = [idx for idx, _ in valid_similarities[:top_n]]
        
        return self.df.iloc[top_indices]
    
    def get_recommendations_by_profile(self, student_profile, top_n=5):
        """
        Krijg aanbevelingen op basis van student profiel

        Parameters:
        -----------
        student_profile : dict
            {'interests': ['data', 'ai'], 'preferred_difficulty': 3, 'level': 'B'}
        """
        eps = 1e-8
        n = len(self.df)
        scores = np.zeros(n)

        # Interest matching (keyword counts)
        interest_counts = np.zeros(n, dtype=float)
        if 'interests' in student_profile and student_profile['interests']:
            interests = [s.lower() for s in student_profile['interests']]
            for idx, combined in enumerate(self.df['combined_text'].fillna('').astype(str)):
                cnt = sum(1 for it in interests if it in combined.lower())
                interest_counts[idx] = cnt
        # normalize interest counts 0-1
        ic_min, ic_max = interest_counts.min(), interest_counts.max()
        if ic_max - ic_min > 0:
            ic_norm = (interest_counts - ic_min) / (ic_max - ic_min + eps)
        else:
            ic_norm = interest_counts  # all zeros
        scores += ic_norm * 2.0  # weight for keyword matches (adjustable)

        # Difficulty matching (already roughly 0-1)
        if 'preferred_difficulty' in student_profile:
            pref_diff = student_profile['preferred_difficulty']
            diff_scores = 1 - np.abs(self.df['estimated_difficulty'].fillna(pref_diff).values - pref_diff) / 5
            diff_scores = np.clip(diff_scores, 0, 1)
            scores += diff_scores * 1.0

        # Level matching
        if 'level' in student_profile:
            level_match = (self.df['level'] == student_profile['level']).astype(int).values
            scores += level_match * 1.5

        # Normalize stored interest-match and popularity to 0-1 (global scale)
        pop = self.df['popularity_score'].fillna(0).astype(float).values
        pop_min, pop_max = pop.min(), pop.max()
        pop_norm = (pop - pop_min) / (pop_max - pop_min + eps)

        ims = self.df['interests_match_score'].fillna(0).astype(float).values
        ims_min, ims_max = ims.min(), ims.max()
        ims_norm = (ims - ims_min) / (ims_max - ims_min + eps)

        # Add normalized contributions with original weights
        scores += ims_norm * 0.5
        scores += pop_norm * 0.3

        # Get top N
        top_indices = scores.argsort()[-top_n:][::-1]

        results = self.df.iloc[top_indices].copy()
        results['match_score'] = scores[top_indices]

        return results
    

    def get_cluster_recommendations(self, cluster_id, top_n=5):
        """Krijg top modules uit een specifiek cluster (met normalisatie)"""
        cluster_modules = self.df[self.df['cluster'] == cluster_id].copy()
        
        eps = 1e-8
        pop_min = self.df['popularity_score'].min()
        pop_max = self.df['popularity_score'].max()
        int_min = self.df['interests_match_score'].min()
        int_max = self.df['interests_match_score'].max()
        
        cluster_modules['pop_norm'] = (cluster_modules['popularity_score'] - pop_min) / (pop_max - pop_min + eps)
        cluster_modules['int_norm'] = (cluster_modules['interests_match_score'] - int_min) / (int_max - int_min + eps)
        
        # Gewichten toepassen (aanpasbaar)
        cluster_modules['score'] = cluster_modules['pop_norm'] * 0.5 + cluster_modules['int_norm'] * 0.5
        
        return cluster_modules.nlargest(top_n, 'score')


# Initialiseer recommender
print("\n[2] Recommender initialiseren...")
recommender = VKMRecommender(model, df_full)
print("Recommender ready")

# DEMO 1: Content-based recommendations
print("\n" + "=" * 80)
print("DEMO 1: CONTENT-BASED RECOMMENDATIONS")
print("=" * 80)

# Kies een random module als basis
base_idx = 10
base_module = df_full.iloc[base_idx]

print(f"\n Basis module:")
print(f"   Naam: {base_module['name']}")
print(f"   Level: {base_module['level']} | Credits: {base_module['studycredit']}")
print(f"   Moeilijkheid: {base_module['estimated_difficulty']}/5")
print(f"   Populariteit: {base_module['popularity_score']:.2f}")

print(f"\n Top 5 vergelijkbare modules:")
recommendations = recommender.get_recommendations(base_idx, top_n=5)

for i, (idx, module) in enumerate(recommendations.iterrows(), 1):
    similarity = model['similarity_matrix'][base_idx][idx]
    print(f"\n{i}. {module['name']}")
    print(f"   Similarity: {similarity:.3f}")
    print(f"   Level: {module['level']} | Credits: {module['studycredit']}")
    print(f"   Moeilijkheid: {module['estimated_difficulty']}/5 | Populariteit: {module['popularity_score']:.2f}")

# DEMO 2: Profile-based recommendations
print("\n" + "=" * 80)
print("DEMO 2: PROFILE-BASED RECOMMENDATIONS")
print("=" * 80)

student_profile = {
    'interests': ['data', 'analyse', 'machine learning', 'AI'],
    'preferred_difficulty': 3,
    'level': 'NLQF5'
}

print(f"\n Student Profiel:")
print(f"   Interesses: {', '.join(student_profile['interests'])}")
print(f"   Gewenste moeilijkheid: {student_profile['preferred_difficulty']}/5")
print(f"   Level: {student_profile['level']}")

print(f"\n Top 5 aanbevelingen:")
profile_recs = recommender.get_recommendations_by_profile(student_profile, top_n=5)

for i, (idx, module) in enumerate(profile_recs.iterrows(), 1):
    print(f"\n{i}. {module['name']}")
    print(f"   Match score: {module['match_score']:.2f}")
    print(f"   Level: {module['level']} | Credits: {module['studycredit']}")
    print(f"   Moeilijkheid: {module['estimated_difficulty']}/5 | Populariteit: {module['popularity_score']:.2f}")
    print(f"   Locatie: {module['location']}")

# DEMO 3: Cluster-based recommendations
print("\n" + "=" * 80)
print("DEMO 3: CLUSTER-BASED RECOMMENDATIONS")
print("=" * 80)

cluster_id = 2
print(f"\n Modules uit Cluster {cluster_id}:")

cluster_recs = recommender.get_cluster_recommendations(cluster_id, top_n=5)

for i, (idx, module) in enumerate(cluster_recs.iterrows(), 1):
    print(f"\n{i}. {module['name']}")
    print(f"   Score: {module['score']:.2f}")
    print(f"   Level: {module['level']} | Credits: {module['studycredit']}")
    print(f"   Moeilijkheid: {module['estimated_difficulty']}/5 | Populariteit: {module['popularity_score']:.2f}")

print("\n" + "=" * 80)
print(" INFERENCE COMPLETE")
print("=" * 80)
print("\nHet model kan nu gebruikt worden voor real-time aanbevelingen!")