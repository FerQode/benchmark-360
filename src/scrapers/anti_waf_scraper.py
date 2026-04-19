import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
import time
import os

def scrape_puntonet_stealth():
    """Intenta bypassear Cloudflare Turnstile"""
    options = uc.ChromeOptions()
    options.add_argument('--headless=new')  # Modo headless moderno
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument('--window-size=1920,1080')
    
    driver = uc.Chrome(options=options)
    try:
        print('INICIANDO Accediendo a Celerity...')
        driver.get('https://www.celerity.ec')
        
        # Espera pasiva para resolver Cloudflare
        time.sleep(15)
        
        # Verifica si pasó el challenge
        if 'verify' not in driver.current_url.lower() and 'challenge' not in driver.current_url.lower():
            html = driver.page_source
            os.makedirs('data/raw/puntonet', exist_ok=True)
            with open('data/raw/puntonet/bypass_success.html', 'w', encoding='utf-8') as f:
                f.write(html)
            print('EXITO Bypass exitoso! HTML guardado')
            return True
        else:
            print('ERROR Sigue en challenge de Cloudflare')
            return False
    except Exception as e:
        print(f'ERROR Error durante ejecucion: {e}')
        return False
    finally:
        driver.quit()

if __name__ == '__main__':
    scrape_puntonet_stealth()
