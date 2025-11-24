import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

print("=" * 80)
print("VKM SMART STUDY COACH - STUDENT-TO-MODULE RECOMMENDER")
print("=" * 80)

# Load model
print("\n[1] Model laden...")
with open('vkm_student_recommender_model.pkl', 'rb') as f:
    model = pickle.load(f)

print(f"Model geladen: {model['model_version']}")
print(f"  - {len(model['dataframe'])} modules beschikbaar")
print(f"  - TF-IDF matrix: {model['tfidf_matrix'].shape}")


class StudentToModuleRecommender:
    """AI-gestuurd Student-to-Module Aanbevelingssysteem"""
    
    def __init__(self, model_artifacts):
        self.tfidf_vectorizer = model_artifacts['tfidf_vectorizer']
        self.tfidf_matrix = model_artifacts['tfidf_matrix']
        self.df = model_artifacts['dataframe']
        
    def recommend_for_student(self, student_text, top_n=5, filters=None, explain=True):
        """
        Hybride aanbevelingen: TF-IDF similarity + numerieke feature bonussen
        
        Parameters:
        -----------
        student_text : str
            Vrije tekst van student over interesses
        top_n : int
            Aantal aanbevelingen
        filters : dict
            Optionele filters {'level': 'NLQF5', 'min_credits': 3}
        explain : bool
            Geef uitleg waarom modules passen
            
        Returns:
        --------
        DataFrame met aanbevelingen en match scores
        """
        # STAP 1: TF-IDF Content Similarity (basis)
        student_vector = self.tfidf_vectorizer.transform([student_text.lower()])
        content_similarities = cosine_similarity(student_vector, self.tfidf_matrix)[0]
        
        # STAP 2: Hybride Scoring - Combineer content met numerieke features
        # Initialiseer scores met content similarity (gewicht: 60%)
        scores = content_similarities * 0.6
        
        # Bonus 1: Popularity Score (gewicht: 15%)
        if 'popularity_score' in self.df.columns:
            popularity_normalized = self.df['popularity_score'].fillna(0) / 10.0  # Schaal naar 0-1
            scores += popularity_normalized.values * 0.15
        
        # Bonus 2: Interests Match Score (gewicht: 15%)
        if 'interests_match_score' in self.df.columns:
            interests_normalized = self.df['interests_match_score'].fillna(0) / 10.0
            scores += interests_normalized.values * 0.15
        
        # Bonus 3: Difficulty Penalty/Bonus (gewicht: 10%)
        # Als student expliciet difficulty noemt, match daarop
        student_lower = student_text.lower()
        if any(term in student_lower for term in ['makkelijk', 'eenvoudig', 'beginners']):
            # Beloon lage difficulty
            difficulty_bonus = (5 - self.df['estimated_difficulty'].fillna(3)) / 5.0
            scores += difficulty_bonus.values * 0.10
        elif any(term in student_lower for term in ['uitdagend', 'advanced', 'gevorderd']):
            # Beloon hoge difficulty
            difficulty_bonus = self.df['estimated_difficulty'].fillna(3) / 5.0
            scores += difficulty_bonus.values * 0.10
        
        # STAP 3: Pas filters toe
        valid_indices = np.arange(len(self.df))
        
        if filters:
            mask = np.ones(len(self.df), dtype=bool)
            
            if 'level' in filters:
                mask &= (self.df['level'] == filters['level']).values
            if 'min_credits' in filters:
                mask &= (self.df['studycredit'] >= filters['min_credits']).values
            if 'max_difficulty' in filters:
                mask &= (self.df['estimated_difficulty'] <= filters['max_difficulty']).values
                
            valid_indices = np.where(mask)[0]
        
        # STAP 4: Sorteer op hybride scores
        valid_scores = [(idx, scores[idx]) for idx in valid_indices]
        valid_scores.sort(key=lambda x: x[1], reverse=True)
        
        # STAP 5: Haal top N modules op
        top_indices = [idx for idx, _ in valid_scores[:top_n]]
        results = self.df.iloc[top_indices].copy()
        results['match_score'] = [scores[idx] for idx in top_indices]
        results['content_similarity'] = [content_similarities[idx] for idx in top_indices]
        
        # STAP 6: Voeg uitleg toe
        if explain:
            results['explanation'] = results.apply(
                lambda row: self._explain_match(student_text, row),
                axis=1
            )
        
        return results
    
    def _explain_match(self, student_text, module_row):
        """
        Genereert contextuele uitleg waarom een module past bij de student.
        Verbeterd: veilige checks, duidelijkere logica, geen ontbrekende kolommen.
        """
        student_lower = student_text.lower()
        
        # Veilig ophalen van module-tekst
        module_name = str(module_row.get('name', '')).lower()
        module_desc = str(module_row.get('combined_text', '')).lower()
        
        # Combineer voor efficiënter zoeken
        module_text = f"{module_name} {module_desc}"
        
        # Definieer keyword-categorieën
        tech_keywords = ['data', 'ai', 'programm', 'software', 'technolog', 'python', 
                        'machine learning', 'algoritme', 'ict']
        social_keywords = ['psycholog', 'coach', 'zorg', 'mensen', 'welzijn', 
                        'sociaal', 'maatschappelijk']
        business_keywords = ['bedrijf', 'management', 'marketing', 'ondernemen', 
                            'strategie', 'financiën']
        creative_keywords = ['design', 'creatief', 'kunst', 'media', 'communicatie',
                            'visueel']
        
        # Detecteer interessegebieden
        is_tech = any(kw in student_lower for kw in tech_keywords)
        is_social = any(kw in student_lower for kw in social_keywords)
        is_business = any(kw in student_lower for kw in business_keywords)
        is_creative = any(kw in student_lower for kw in creative_keywords)
        
        # Detecteer module-focus
        module_tech = any(kw in module_text for kw in tech_keywords)
        module_social = any(kw in module_text for kw in social_keywords)
        module_business = any(kw in module_text for kw in business_keywords)
        module_creative = any(kw in module_text for kw in creative_keywords)
        
        # Genereer contextuele uitleg (match eerst op domein)
        if is_tech and module_tech:
            return "Past bij je interesse in technologie en data-analyse"
        elif is_social and module_social:
            return "Sluit aan bij je motivatie om met mensen te werken"
        elif is_business and module_business:
            return "Bereidt je voor op zakelijke en strategische rollen"
        elif is_creative and module_creative:
            return "Ontwikkelt je creatieve en communicatieve vaardigheden"
        
        # Fallback op numerieke features (veilig met .get())
        popularity = module_row.get('popularity_score', 0)
        difficulty = module_row.get('estimated_difficulty', 3)
        
        if popularity > 7:
            return "Populaire module die goed aansluit bij je profiel"
        elif difficulty <= 2:
            return "Toegankelijke module om je interesses te verkennen"
        elif difficulty >= 4:
            return "Uitdagende module voor ambitieuze studenten"
        else:
            level = module_row.get('level', 'onbekend')
            return f"Relevante {level}-module voor jouw ontwikkeling"
    
    def batch_recommend(self, student_texts, top_n=3):
        """Geef aanbevelingen voor meerdere studenten tegelijk"""
        results = {}
        for i, text in enumerate(student_texts):
            results[f"Student_{i+1}"] = self.recommend_for_student(text, top_n=top_n, explain=False)
        return results


# Initialiseer recommender
print("\n[2] Recommender initialiseren...")
recommender = StudentToModuleRecommender(model)
print("Recommender gereed!")

# DEMO 1: Student met interesse in data & AI
print("\n" + "=" * 80)
print("DEMO 1: STUDENT GEÏNTERESSEERD IN DATA & AI")
print("=" * 80)

student_1 = """
Ik ben geïnteresseerd in data analyse, machine learning en kunstmatige intelligentie.
Ik wil graag leren programmeren in Python en werken met datasets.
"""

print(f"\nStudentprofiel:\n{student_1.strip()}")
print("\n Top 5 aanbevelingen:")

recommendations_1 = recommender.recommend_for_student(student_1, top_n=5)

for i, (idx, module) in enumerate(recommendations_1.iterrows(), 1):
    print(f"\n{i}. {module['name']}")
    print(f"   Match score: {module['match_score']:.3f}")
    print(f"   Level: {module['level']} | Credits: {module['studycredit']} | Locatie: {module['location']}")
    print(f"   Moeilijkheid: {module['estimated_difficulty']:.1f}/5")
    print(f"   → {module['explanation']}")
    desc = str(module.get('shortdescription', 'Geen beschrijving'))[:100]
    print(f"   Omschrijving: {desc}...")

# DEMO 2: Student met interesse in psychologie & coaching
print("\n" + "=" * 80)
print("DEMO 2: STUDENT GEÏNTERESSEERD IN PSYCHOLOGIE & COACHING")
print("=" * 80)

student_2 = """
Ik ben geïnteresseerd in psychologie, coaching en zorg.
Ik wil graag werken met mensen en hen helpen groeien.
"""

print(f"\nStudentprofiel:\n{student_2.strip()}")
print("\n Top 5 aanbevelingen:")

recommendations_2 = recommender.recommend_for_student(student_2, top_n=5)

for i, (idx, module) in enumerate(recommendations_2.iterrows(), 1):
    print(f"\n{i}. {module['name']}")
    print(f"   Match score: {module['match_score']:.3f}")
    print(f"   Level: {module['level']} | Credits: {module['studycredit']}")
    print(f"   → {module['explanation']}")
