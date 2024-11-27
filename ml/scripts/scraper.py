import praw
import requests
import pytesseract
from PIL import Image
from io import BytesIO
import pandas as pd
import time
from tqdm import tqdm
import os
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

class LinkedInPostScraper:
    def __init__(self):
        self.reddit = praw.Reddit(
            client_id="hUmD_ce2Qde-k1eM9NZi0g",
            client_secret="75uOrEH4papSqxhkbvODLcrrOEXPEA",
            user_agent="Exciting_Bedroom6074"
        )
        
        os.makedirs('data/images', exist_ok=True)

    def download_image(self, url, post_id):
        """Download image from URL and save locally"""
        try:
            response = requests.get(url)
            if response.status_code == 200:
                img_path = f'data/images/{post_id}.jpg'
                with open(img_path, 'wb') as f:
                    f.write(response.content)
                return img_path
            return None
        except Exception as e:
            print(f"Error downloading image: {e}")
            return None

    def extract_text_from_image(self, image_path):
        """Extract text from image using OCR"""
        try:
            # Load image
            image = Image.open(image_path)
            
            # OCR configuration for better LinkedIn post extraction
            custom_config = r'--oem 3 --psm 6 -l eng'
            
            # Extract text
            text = pytesseract.image_to_string(image, config=custom_config)
            
            # Clean extracted text
            text = self.clean_text(text)
            
            return text
        except Exception as e:
            print(f"Error extracting text: {e}")
            return None

    def clean_text(self, text):
        """Clean and normalize extracted text"""
        if not text:
            return ""
            
        text = text.replace('|', 'I')
        text = ' '.join(text.split())
        
        ui_elements = [
            "Like", "Comment", "Share", "Reply",
            "• 1st", "• 2nd", "• 3rd",
            "reactions", "comments", "Comment as",
            "See translation", "Edited"
        ]
        
        for element in ui_elements:
            text = text.replace(element, '')
            
        return text.strip()

    def scrape_posts(self, limit=10000):
        """Scrape posts from r/LinkedInLunatics"""
        posts_data = []
        subreddit = self.reddit.subreddit('LinkedInLunatics')
        
        print(f"Scraping {limit} posts from r/LinkedInLunatics...")
        
        count = 0
        for post in tqdm(subreddit.hot(limit=limit)):
            if hasattr(post, 'url') and any(post.url.endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
                image_path = self.download_image(post.url, post.id)
                if image_path:
                    text = self.extract_text_from_image(image_path)
                    if text and len(text) > 50:
                        print(post.id, count)
                        posts_data.append({
                            'post_id': post.id,
                            'title': post.title,
                            'score': post.score,
                            'url': post.url,
                            'text': text,
                            'num_comments': post.num_comments,
                            'upvote_ratio': post.upvote_ratio
                        })
                        count += 1
                time.sleep(2)

        df = pd.DataFrame(posts_data)
        df.to_csv('data/linkedin_posts.csv', index=False)
        print(f"Scraped {len(posts_data)} posts successfully!")
        return df

    def calculate_cringe_score(self, row):
        """Calculate initial cringe score based on Reddit metrics"""
        base_score = min((row['score'] / 100) * 20 + (row['num_comments'] / 10) * 20, 100)
        
        ratio_factor = (1 - row['upvote_ratio']) * 20

        cringe_score = min(base_score + ratio_factor, 100)
        return round(cringe_score, 2)

    def prepare_training_data(self):
        """Prepare final dataset for training"""
        df = pd.read_csv('data/linkedin_posts.csv')
        
        df['cringe_score'] = df.apply(self.calculate_cringe_score, axis=1)
        
        df.to_csv('data/training_data.csv', index=False)
        print(f"Prepared training data with {len(df)} samples!")
        
        return df

def main():
    scraper = LinkedInPostScraper()
    
    # Scrape posts
    df = scraper.scrape_posts(limit=10000)
    print(df)
    
    # Prepare training data
    #training_data = scraper.prepare_training_data()
    
    #print("\nSample of collected data:")
    #print(training_data[['text', 'cringe_score']].head())
    
    #print("\nStats of cringe scores:")
    #print(training_data['cringe_score'].describe())

if __name__ == "__main__":
    main()