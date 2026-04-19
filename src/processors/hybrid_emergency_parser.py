import re
from bs4 import BeautifulSoup
from typing import Dict, List, Optional

class HybridEmergencyParser:
    @staticmethod
    def extract_from_claro(html: str) -> List[Dict]:
        """Extrae planes de Claro usando SOLO regex (sin LLM)"""
        soup = BeautifulSoup(html, 'html.parser')
        planes = []
        
        # Patrón específico de Claro Ecuador
        price_pattern = r'\$?(\d{1,3}[\.\,]\d{2})'
        speed_pattern = r'(\d+)\s*(?:Mbps|Megas|MB)'
        
        for plan_card in soup.find_all('div', class_=re.compile('plan|card|offer', re.I)):
            text = plan_card.get_text()
            
            price_match = re.search(price_pattern, text)
            speed_match = re.search(speed_pattern, text)
            
            if price_match and speed_match:
                planes.append({
                    'precio': float(price_match.group(1).replace(',', '.')),
                    'velocidad': int(speed_match.group(1)),
                    'nombre': text[:50].strip().replace('\n', ' ')
                })
        
        return planes

    @staticmethod
    def extract_from_netlife(html: str) -> List[Dict]:
        """Extrae planes de Netlife usando SOLO regex (sin LLM)"""
        soup = BeautifulSoup(html, 'html.parser')
        planes = []
        
        # Patrones específicos de Netlife
        price_pattern = r'\$?(\d{1,3}[\.\,]\d{2})'
        speed_pattern = r'(\d+)\s*(?:Mbps|Megas|MB)'
        
        # Netlife usa cards con clase 'plan-card' o 'offer-card' u otras
        for plan_card in soup.find_all('div', class_=re.compile('plan|card|offer', re.I)):
            text = plan_card.get_text()
            
            # Buscamos precio
            price_tag = plan_card.find('span', class_=re.compile('price|valor', re.I))
            if price_tag:
                price_match = re.search(price_pattern, price_tag.text)
            else:
                price_match = re.search(price_pattern, text)
            
            speed_match = re.search(speed_pattern, text)
            
            if price_match and speed_match:
                nombre_tag = plan_card.find('h3') or plan_card.find('h2')
                nombre = nombre_tag.text.strip() if nombre_tag else text[:50].strip().replace('\n', ' ')
                
                planes.append({
                    'nombre_plan': nombre,
                    'precio_plan': float(price_match.group(1).replace(',', '.')),
                    'velocidad_download_mbps': int(speed_match.group(1)),
                    'velocidad_upload_mbps': int(speed_match.group(1)),
                    'marca': 'Netlife',
                    'tecnologia': 'fibra_optica'
                })
        
        return planes

    @classmethod
    def extract_from_any(cls, html: str, isp_name: str) -> List[Dict]:
        isp_lower = isp_name.lower()
        if 'claro' in isp_lower:
            return cls.extract_from_claro(html)
        elif 'netlife' in isp_lower:
            return cls.extract_from_netlife(html)
        else:
            return []  # Fallback a LLM para otros ISPs
