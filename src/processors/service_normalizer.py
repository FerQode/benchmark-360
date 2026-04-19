from rapidfuzz import fuzz
from typing import Dict

class ServiceNormalizer:
    SERVICE_MAP = {
        'disney_plus': ['disney+', 'disney plus', 'disney+ premium', 'disney basic'],
        'hbo_max': ['max', 'hbo max', 'hbo+', 'warner', 'max premium'],
        'netflix': ['netflix', 'netflix standard', 'netflix premium'],
        'amazon_prime': ['prime video', 'amazon prime', 'prime video'],
        'paramount_plus': ['paramount+', 'paramount plus', 'paramount'],
    }
    
    @classmethod
    def normalize(cls, raw_name: str) -> str:
        raw_lower = raw_name.lower().strip()
        
        # Búsqueda difusa
        best_match = None
        best_score = 0
        
        for standard, variants in cls.SERVICE_MAP.items():
            for variant in variants:
                score = fuzz.ratio(raw_lower, variant)
                if score > best_score and score > 80:
                    best_score = score
                    best_match = standard
        
        return best_match if best_match else raw_lower.replace(' ', '_')

# Prueba
if __name__ == '__main__':
    tests = ['Disney+', 'HBO Max', 'netflix standard', 'Prime Video', 'Disney Plus Premium']
    for test in tests:
        result = ServiceNormalizer.normalize(test)
        print(f'{test:20} -> {result}')
