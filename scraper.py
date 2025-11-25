"""
Reference Image Scraper - Deployment Version
Based on proven working Selenium code
"""

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import os
import requests
import random
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ReferenceImageScraper:
    def __init__(self, output_dir='reference_images', headless=False):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        chrome_options = Options()
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--no-sandbox")
        if headless:
            chrome_options.add_argument("--headless")
        
        self.driver = webdriver.Chrome(options=chrome_options)
        logger.info("Chrome driver initialized")
    
    def load_metadata(self, metadata_path):
        """Load metadata and extract ASINs with product names"""
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        asin_products = {}
        
        if 'BIN_FCSKU_DATA' in metadata:
            bin_data = metadata['BIN_FCSKU_DATA']
            
            for key, value in bin_data.items():
                asin = value.get('asin')
                name = value.get('name', '')
                if asin and name:
                    asin_products[asin] = name
        
        logger.info(f"Found {len(asin_products)} unique ASINs in metadata")
        return asin_products
    
    def scrape_image(self, asin, product_name):
        """Scrape image using proven method - saves flat structure"""
        try:
            # Save directly to output folder (no subfolder)
            img_path = os.path.join(self.output_dir, f"{asin}.jpg")
            
            # Skip if already exists
            if os.path.exists(img_path):
                logger.info(f"Skipping {asin} - already exists")
                return True
            
            # Search Google Images
            search_query = product_name
            google_url = f"https://www.google.com/search?tbm=isch&q={search_query}"
            
            self.driver.get(google_url)
            time.sleep(random.uniform(3, 5))
            
            # Find clickable images
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div[data-id]"))
            )
            
            image_containers = self.driver.find_elements(By.CSS_SELECTOR, "div[data-id]")
            
            if len(image_containers) == 0:
                logger.warning(f"No images found for {asin}")
                return False
            
            download_count = 0
            
            # Try first 5 images
            for img_idx in range(min(5, len(image_containers))):
                try:
                    self.driver.execute_script("arguments[0].scrollIntoView();", 
                                             image_containers[img_idx])
                    time.sleep(0.5)
                    
                    image_containers[img_idx].click()
                    time.sleep(2)
                    
                    large_images = self.driver.find_elements(By.CSS_SELECTOR, "img[src*='http']")
                    
                    for large_img in large_images:
                        img_url = large_img.get_attribute('src')
                        
                        if not img_url or 'data:image' in img_url or 'gstatic' in img_url:
                            continue
                        
                        if len(img_url) < 30:
                            continue
                        
                        img_response = requests.get(img_url, timeout=10)
                        
                        if img_response.status_code == 200 and len(img_response.content) > 5000:
                            # Save directly (no subfolder)
                            with open(img_path, 'wb') as f:
                                f.write(img_response.content)
                            download_count += 1
                            break
                    
                    if download_count >= 1:
                        break
                        
                except:
                    continue
            
            if download_count > 0:
                logger.info(f"Saved: {asin}")
                return True
            else:
                logger.warning(f"Failed: {asin}")
                return False
            
        except Exception as e:
            logger.error(f"Error scraping {asin}: {str(e)}")
            return False
    
    def scrape_from_metadata(self, metadata_path):
        """Main method: Load metadata and scrape all reference images"""
        asin_products = self.load_metadata(metadata_path)
        
        results = {}
        total = len(asin_products)
        success_count = 0
        
        for idx, (asin, product_name) in enumerate(asin_products.items(), 1):
            logger.info(f"Processing {idx}/{total}: {asin}")
            
            success = self.scrape_image(asin, product_name)
            results[asin] = success
            
            if success:
                success_count += 1
            
            # Random delay between requests
            if idx < total:
                delay = random.randint(8, 15)
                time.sleep(delay)
        
        logger.info(f"Successfully scraped {success_count}/{total} images")
        return results
    
    def close(self):
        """Close browser"""
        try:
            self.driver.quit()
            logger.info("Browser closed")
        except:
            pass


# STANDALONE FUNCTION FOR STREAMLIT
def scrape_reference_images(asins_dict, output_folder):
    """
    Scrape reference images for given ASINs using Selenium
    
    Args:
        asins_dict: Dict of {ASIN: product_name}
        output_folder: Where to save images (flat structure)
    
    Returns:
        Number of successfully scraped images
    """
    scraper = ReferenceImageScraper(output_dir=output_folder, headless=True)
    
    try:
        scraped_count = 0
        total = len(asins_dict)
        
        for idx, (asin, product_name) in enumerate(asins_dict.items(), 1):
            logger.info(f"Scraping {idx}/{total}: {asin}")
            
            success = scraper.scrape_image(asin, product_name)
            
            if success:
                scraped_count += 1
            
            # Delay between requests
            if idx < total:
                time.sleep(random.randint(3, 6))
        
        logger.info(f"Successfully scraped {scraped_count}/{total} images")
        return scraped_count
        
    finally:
        scraper.close()


def main():
    """Main execution"""
    
    METADATA_PATH = r'C:\Users\Hrida\OneDrive\Desktop\Applied AI\Assignment-1\data-main\metadata\00312.json'
    
    scraper = ReferenceImageScraper(
        output_dir='reference_images',
        headless=True
    )
    
    try:
        results = scraper.scrape_from_metadata(METADATA_PATH)
        
        print("\n" + "="*70)
        print("SCRAPING COMPLETE")
        print("="*70)
        print(f"Images saved to: reference_images/")
        
    finally:
        scraper.close()


if __name__ == "__main__":
    main()