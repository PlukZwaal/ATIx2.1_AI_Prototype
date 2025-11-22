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
        Geeft module-aanbevelingen voor een student op basis van hun interesses.
        
        Parameters:
        -----------
        student_text : str
            Vrije tekst van de student over hun interesses (bijv. "Ik wil leren over data en AI")
        top_n : int
            Aantal aanbevelingen
        filters : dict
            Optionele filters zoals {'level': 'NLQF5', 'min_credits': 3}
        explain : bool
            Geef uitleg waarom modules passen bij de student
            
        Returns:
        --------
        DataFrame met aanbevelingen en match scores
        """
        # STAP 1: Vectoriseer het studentprofiel met DEZELFDE TF-IDF vectorizer
        student_vector = self.tfidf_vectorizer.transform([student_text.lower()])
        
        # STAP 2: Bereken cosine similarity tussen student en alle modules
        similarities = cosine_similarity(student_vector, self.tfidf_matrix)[0]
        
        # STAP 3: Pas filters toe indien opgegeven
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
        
        # STAP 4: Sorteer op similarity score
        valid_similarities = [(idx, similarities[idx]) for idx in valid_indices]
        valid_similarities.sort(key=lambda x: x[1], reverse=True)
        
        # STAP 5: Haal top N modules op
        top_indices = [idx for idx, _ in valid_similarities[:top_n]]
        results = self.df.iloc[top_indices].copy()
        results['match_score'] = [similarities[idx] for idx in top_indices]
        
        # STAP 6: Voeg uitleg toe (waarom past deze module?)
        if explain:
            results['explanation'] = results.apply(
                lambda row: self._explain_match(student_text, row, student_vector),
                axis=1
            )
        
        return results
    
    def _explain_match(self, student_text, module_row, student_vector):
        """Genereert intelligente uitleg waarom een module past bij de student"""
        
        # Categoriseer het type match op basis van context
        student_lower = student_text.lower()
        module_desc = module_row['combined_text'].lower()
        module_name = module_row['name'].lower()
        
        # Detecteer interessegebieden in studentprofiel
        is_tech = any(term in student_lower for term in ['data', 'ai', 'programm', 'software', 'technolog', 'python', 'machine learning'])
        is_social = any(term in student_lower for term in ['psycholog', 'coach', 'zorg', 'mensen', 'welzijn', 'sociaal'])
        is_business = any(term in student_lower for term in ['bedrijf', 'management', 'marketing', 'ondernemen', 'strategie'])
        is_creative = any(term in student_lower for term in ['design', 'creatief', 'kunst', 'media', 'communicatie'])
        
        # Detecteer overeenkomstige focus in module
        module_tech = any(term in module_desc or term in module_name for term in ['data', 'ai', 'software', 'programm', 'ict', 'technolog'])
        module_social = any(term in module_desc or term in module_name for term in ['psycholog', 'coach', 'zorg', 'welzijn', 'sociaal'])
        module_business = any(term in module_desc or term in module_name for term in ['bedrijf', 'management', 'marketing', 'ondernemen'])
        module_creative = any(term in module_desc or term in module_name for term in ['design', 'creatief', 'communicatie', 'media'])
        
        # Genereer contextuele uitleg
        if is_tech and module_tech:
            return "Past bij je interesse in technologie en data-analyse"
        elif is_social and module_social:
            return "Sluit aan bij je motivatie om met mensen te werken"
        elif is_business and module_business:
            return "Bereidt je voor op zakelijke en strategische rollen"
        elif is_creative and module_creative:
            return "Ontwikkelt je creatieve en communicatieve vaardigheden"
        elif module_row['popularity_score'] > 7:
            return "Populaire module die goed aansluit bij je profiel"
        elif module_row['estimated_difficulty'] <= 3:
            return "Toegankelijke module om je interesses te verkennen"
        else:
            return f"Relevante {module_row['level']} module voor jouw ontwikkeling"
    
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
    if 'shortdescription' in module:
        desc = str(module['shortdescription'])[:100]
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