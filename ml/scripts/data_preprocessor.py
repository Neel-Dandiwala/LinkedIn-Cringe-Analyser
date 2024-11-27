import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from textblob import TextBlob
import re
import emoji
from collections import Counter

class LinkedInDataPreprocessor:
    def __init__(self):
        self.corporate_buzzwords = set([ 'b2b sales',
            'synergy', 'leverage', 'saas', 'paradigm', 'disrupt', 'innovation',
            'thought leadership', 'bandwidth', 'circle back', 'deep dive',
            'rockstar', 'ninja', 'guru', 'crushing it', 'hustle', 'grind'
        ])
        
        self.humble_brag_phrases = set([
            'god', 'humble', 'blessed', 'honored', 'grateful', 'thankful',
            'privileged', 'excited to announce', 'proud to share', 'exciting news'
        ])

    def extract_features(self, text):
        """Extract relevant features from text"""
        features = {}
        
        features['length'] = len(text)
        features['word_count'] = len(text.split())
        features['avg_word_length'] = np.mean([len(word) for word in text.split()])
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['hashtag_count'] = text.count('#')
        features['emoji_count'] = len([c for c in text if c in emoji.EMOJI_DATA])
        features['caps_ratio'] = sum(1 for c in text if c.isupper()) / len(text)
        features['all_caps_words'] = len([w for w in text.split() if w.isupper()])
        features['buzzword_count'] = sum(1 for word in text.lower().split() if word in self.corporate_buzzwords)
        features['humble_brag_count'] = sum(1 for phrase in self.humble_brag_phrases if phrase in text.lower())
        blob = TextBlob(text)
        features['sentiment_polarity'] = blob.sentiment.polarity
        features['sentiment_subjectivity'] = blob.sentiment.subjectivity
        
        return features

    def preprocess_dataset(self, csv_path):
        """Preprocess the entire dataset"""
        df = pd.read_csv(csv_path)
        
        features_list = []
        for text in df['text']:
            features = self.extract_features(text)
            features_list.append(features)
            
        features_df = pd.DataFrame(features_list)
        
        features_df['cringe_score'] = df['cringe_score']
        
        features_df = features_df.fillna(0)
        
        features_df.to_csv('data/processed_features.csv', index=False)
        print(f"Processed {len(features_df)} posts!")
        
        return features_df

    def prepare_train_test_split(self, features_df, test_size=0.2):
        """Prepare training and test datasets"""
        X = features_df.drop('cringe_score', axis=1)
        y = features_df['cringe_score']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        return X_train, X_test, y_train, y_test

def main():
    preprocessor = LinkedInDataPreprocessor()
    
    features_df = preprocessor.preprocess_dataset('data/training_data.csv')
    
    X_train, X_test, y_train, y_test = preprocessor.prepare_train_test_split(features_df)
    
    print("\nFeature statistics:")
    print(features_df.describe())
    
    print("\nFeature correlations with cringe score:")
    correlations = features_df.corr()['cringe_score'].sort_values(ascending=False)
    print(correlations)

if __name__ == "__main__":
    main()