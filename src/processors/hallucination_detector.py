# src/processors/hallucination_detector.py

import re
from typing import Dict, Any

class HallucinationDetector:
    """
    Detector de alucinaciones semánticas en la extracción de planes de ISPs.
    Detecta si el LLM cruzó cables e inventó planes que pertenecen a otro proveedor.
    """

    # Lista de los principales competidores en Ecuador para cruzar nombres
    COMPETITORS = [
        'claro', 'netlife', 'xtrim', 'cnt', 'puntonet', 'celerity', 'alfanet',
        'hughesnet', 'starlink', 'dfibra'
    ]

    @classmethod
    def detect(cls, plan: Dict[str, Any], isp_key: str) -> bool:
        """
        Devuelve True si el plan parece ser una alucinación (ej: nombre de plan Claro para Netlife)
        """
        nombre = plan.get('nombre_plan', '').lower()
        isp_lower = isp_key.lower()

        # Extraer palabras clave del nombre del plan
        for competitor in cls.COMPETITORS:
            if competitor != isp_lower and competitor in nombre:
                # Si el ISP es celerity y aparece puntonet, no es alucinación porque son la misma empresa
                if (isp_lower == 'celerity' and competitor == 'puntonet') or \
                   (isp_lower == 'puntonet' and competitor == 'celerity'):
                    continue
                return True
        return False

    @classmethod
    def get_reason(cls, plan: Dict[str, Any], isp_key: str) -> str:
        """
        Devuelve la razón detallada de la alucinación
        """
        nombre = plan.get('nombre_plan', '').lower()
        isp_lower = isp_key.lower()

        for competitor in cls.COMPETITORS:
            if competitor != isp_lower and competitor in nombre:
                if (isp_lower == 'celerity' and competitor == 'puntonet') or \
                   (isp_lower == 'puntonet' and competitor == 'celerity'):
                    continue
                return f"Plan de '{isp_key}' contiene nombre de competidor '{competitor}'"
        
        return "Alucinación desconocida"
