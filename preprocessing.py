
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Example usage
if __name__ == "__main__":
    df = pd.read_csv('sample_educational_dataset.csv')
    df['Processed_Description'] = df['Description'].apply(preprocess_text)
    df.to_csv('preprocessed_dataset.csv', index=False)
    print("Preprocessing complete. Output saved to preprocessed_dataset.csv")
