import pandas as pd
from textblob import TextBlob
from bs4 import BeautifulSoup
import re
import emoji
import unicodedata
import spacy

nlp = spacy.load('en_core_web_sm')

class DataPreprocessor:
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

    def clean_text(self, text):
        """Clean the text by removing emojis and URLs"""
        text = BeautifulSoup(text, 'html.parser').get_text()
        text = unicodedata.normalize('NFKD', text)
        emojis = [c for c in text if c in emoji.EMOJI_DATA]
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # URLs
        text = re.sub(r'\S+@\S+', '', text) # Emails
        text = re.sub(r'[^\x00-\x7F]+', '', text) # Non-ASCII
        doc = nlp(text)

        cleaned_tokens = []
        for token in doc:
            if token.like_num or token.text in ['.', '!', '?']:
                cleaned_tokens.append(token.text)
            elif not token.is_punct and not token.is_space:
                cleaned_tokens.append(token.text.lower())

        text = ' '.join(cleaned_tokens)

        ui_elements = ['Following', 'Connect', '• \d+', 'comments?', 'likes?', 'Repost', 'Send', 'View profile']
        for element in ui_elements:
            text = re.sub(element, '', text, flags=re.IGNORECASE)
        
        text = re.sub(r'\b[a-zA-Z]\b', '', text)

        text = str(TextBlob(text).correct())

        text = text + ' ' + ' '.join(emojis)
        text = ' '.join(text.split())

        return text

    def extract_features(self, text):
        """Extract relevant features from text"""
        doc = nlp(text)
        features = {}
        
        features['length'] = len(text)
        features['emoji_count'] = len([c for c in text if c in emoji.EMOJI_DATA])
        features['hashtag_count'] = len([token.text for token in doc if token.text.startswith('#')])
        features['exclamation_count'] = len([token.text for token in doc if token.text == '!'])
        features['question_count'] = len([token.text for token in doc if token.text == '?'])
        features['capitalized_word_count'] = len([word for word in text.split() if word[0].isupper()])
        features['sentence_count'] = len(list(doc.sents))
        features['avg_word_length'] = sum(len(token.text) for token in doc if token.is_alpha) / len([token for token in doc if token.is_alpha])
        features['verb_count'] = len([token for token in doc if token.pos_ == 'VERB'])
        features['noun_count'] = len([token for token in doc if token.pos_ == 'NOUN'])
        features['adjective_count'] = len([token for token in doc if token.pos_ == 'ADJ'])
        features['entity_count'] = len(doc.ents)
        features['has_numerical_list'] = 1 if re.search(r'\d+\.\s', text) else 0

        features['word_count'] = len(text.split())
        features['buzzword_count'] = sum(1 for word in text.lower().split() if word in self.corporate_buzzwords)
        features['buzzword_ratio'] = sum(1 for word in text.lower().split() if word in self.corporate_buzzwords) / max(1, len(text.split()))
        features['humble_brag_count'] = sum(1 for phrase in self.humble_brag_phrases if phrase in text.lower())
        features['personal_pronouns'] = len([token for token in doc if token.pos_ == 'PRON' and token.text.lower() in ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']])
        features['motivation_score'] = sum(1 for word in text.lower().split() if word in ['success', 'achieve', 'accomplish', 'succeed', 'overcome', 'thrive', 'flourish', 'excel', 'surpass', 'transcend'])
        features['excessive_punctuation'] = 1 if re.search(r'[!?][2,]', text) else 0
        features['has_call_to_action'] = 1 if re.search(r'like|follow|connect|message|learn more|click|visit|check out|explore|discover|get in touch|get started|join|sign up|start|begin|get involved|engage|participate|support|donate|contribute|participate|engage|support|agree', text, 
        flags=re.IGNORECASE) else 0
        features['starts_with_number'] = 1 if re.match(r'^\d', text.strip()) else 0
        features['contains_money_reference'] = 1 if re.search(r'\b\d+(?:\.\d{1,2})?\b', text) else 0

        blob = TextBlob(text)
        features['sentiment_polarity'] = blob.sentiment.polarity
        features['sentiment_subjectivity'] = blob.sentiment.subjectivity
        features['is_extemely_positive'] = 1 if features['sentiment_polarity'] > 0.9 else 0
        
        return features

    def preprocess_dataset(self, csv_path):
        """Preprocess the entire dataset"""
        df = pd.read_csv(csv_path)
        
        features_list = []
        for text in df['text']:
            cleaned_text = self.clean_text(text)
            features = self.extract_features(cleaned_text)
            features_list.append(features)
            
        features_df = pd.DataFrame(features_list)
        
        features_df['cringe_score'] = df['score']
        
        features_df = features_df.fillna(0)
        
        features_df.to_csv('data/processed_features.csv', index=False)
        print(f"Processed {len(features_df)} posts!")
        
        return features_df

def main():
    preprocessor = DataPreprocessor()
    features_df = preprocessor.preprocess_dataset('data/linkedin_posts.csv')
    
    print("\nFeature statistics:")
    print(features_df.describe())
    
    print("\nFeature correlations with cringe score:")
    correlations = features_df.corr()['cringe_score'].sort_values(ascending=False)
    print(correlations)


if __name__ == "__main__":
    main()