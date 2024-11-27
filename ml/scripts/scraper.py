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

class LinkedInPostScraper:
    def __init__(self):
        # Initialize Reddit API client
        self.reddit = praw.Reddit(
            client_id="hUmD_ce2Qde-k1eM9NZi0g",
            client_secret="75uOrEH4papSqxhkbvODLcrrOEXPEA",
            user_agent="Exciting_Bedroom6074"
        )
        
        # Create directories for data
        os.makedirs('data/images', exist_ok=True)
        os.makedirs('data/text', exist_ok=True)

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
            
        # Remove common OCR artifacts
        text = text.replace('|', 'I')
        text = ' '.join(text.split())
        
        # Remove common LinkedIn UI elements
        ui_elements = [
            "Like", "Comment", "Share", "Reply",
            "• 1st", "• 2nd", "• 3rd",
            "reactions", "comments", "Comment as",
            "See translation", "Edited"
        ]
        
        for element in ui_elements:
            text = text.replace(element, '')
            
        return text.strip()

    def scrape_posts(self, limit=10):
        """Scrape posts from r/LinkedInLunatics"""
        posts_data = []
        subreddit = self.reddit.subreddit('LinkedInLunatics')
        print(subreddit)
        
        print(f"Scraping {limit} posts from r/LinkedInLunatics...")
        
        for post in tqdm(subreddit.hot(limit=limit)):
            # Check if post has an image
            if hasattr(post, 'url') and any(post.url.endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
                # Download image
                image_path = self.download_image(post.url, post.id)
                if image_path:
                    # Extract text
                    text = self.extract_text_from_image(image_path)
                    if text and len(text) > 50:  # Ensure we have meaningful text
                        posts_data.append({
                            'post_id': post.id,
                            'title': post.title,
                            'score': post.score,
                            'url': post.url,
                            'text': text,
                            'num_comments': post.num_comments,
                            'upvote_ratio': post.upvote_ratio
                        })
                        
                        # Save text to file
                        with open(f'data/text/{post.id}.txt', 'w', encoding='utf-8') as f:
                            f.write(text)
                
                # Reddit API rate limiting
                time.sleep(2)

        # Save to DataFrame
        df = pd.DataFrame(posts_data)
        df.to_csv('data/linkedin_posts.csv', index=False)
        print(f"Scraped {len(posts_data)} posts successfully!")
        return df

    def calculate_cringe_score(self, row):
        """Calculate initial cringe score based on Reddit metrics"""
        # Higher scores for more upvoted and commented posts
        base_score = min((row['score'] / 100) * 20 + (row['num_comments'] / 10) * 20, 100)
        
        # Adjust based on upvote ratio (controversial posts might be cringier)
        ratio_factor = (1 - row['upvote_ratio']) * 20
        
        # Final score
        cringe_score = min(base_score + ratio_factor, 100)
        return round(cringe_score, 2)

    def prepare_training_data(self):
        """Prepare final dataset for training"""
        df = pd.read_csv('data/linkedin_posts.csv')
        
        # Calculate cringe scores
        df['cringe_score'] = df.apply(self.calculate_cringe_score, axis=1)
        
        # Save training data
        df.to_csv('data/training_data.csv', index=False)
        print(f"Prepared training data with {len(df)} samples!")
        
        return df

def main():
    scraper = LinkedInPostScraper()
    
    # Scrape posts
    df = scraper.scrape_posts(limit=1000)
    print(df)
    
    # Prepare training data
    #training_data = scraper.prepare_training_data()
    
    #print("\nSample of collected data:")
    #print(training_data[['text', 'cringe_score']].head())
    
    #print("\nStats of cringe scores:")
    #print(training_data['cringe_score'].describe())

if __name__ == "__main__":
    main()