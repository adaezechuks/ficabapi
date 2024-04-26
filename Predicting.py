import xml.etree.ElementTree as ET
import re
import nltk
import spacy
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('wordnet')

class Predicting:
    def __init__(self):
        # Load SpaCy's English NLP model
        self.nlp = spacy.load("en_core_web_sm")
        self.model = None

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = word_tokenize(text)
        lemmatizer = WordNetLemmatizer()
        return ' '.join([lemmatizer.lemmatize(token) for token in tokens])

    def extract_aspects(self, text):
        doc = self.nlp(text)
        aspects = set()
        for token in doc:
            if token.pos_ == 'NOUN' and token.dep_ in ['compound', 'nsubj', 'dobj', 'attr', 'pobj']:
                aspects.add(token.text)
        return list(aspects)

    def load_or_train_model(self, xml_file=None):
        try:
            with open('svm_pipeline.pkl', 'rb') as model_file:
                self.model = pickle.load(model_file)
        except FileNotFoundError:
            if not xml_file:
                raise ValueError("Training file must be provided for initial training.")
            data = self.parse_xml_to_dataset(xml_file)
            self.train_and_save_model(data)

    def parse_xml_to_dataset(self, xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        data = []
        for review in root.findall('Review'):
            sentences = review.find('sentences')
            for sentence in sentences.findall('sentence'):
                text = sentence.find('text').text
                opinions = sentence.find('Opinions')
                if opinions is not None:
                    aspects = self.extract_aspects(text)
                    for aspect in aspects:
                        for opinion in opinions.findall('Opinion'):
                            polarity = opinion.get('polarity')
                            preprocessed_text = self.preprocess_text(text)
                            data.append((preprocessed_text + " " + aspect, polarity))
        return data

    def train_and_save_model(self, data):
        texts = [d[0] for d in data]
        labels = [d[1] for d in data]
        X_train, _, y_train, _ = train_test_split(texts, labels, test_size=0.2, random_state=42)

        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('svm', SVC(kernel='linear', class_weight='balanced'))
        ])

        param_grid = {
            'tfidf__max_features': [500, 1000, 5000],
            'svm__C': [0.1, 1, 10]
        }

        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)

        with open('svm_pipeline.pkl', 'wb') as model_file:
            pickle.dump(grid_search.best_estimator_, model_file)
        self.model = grid_search.best_estimator_

    def predict_opinion(self, review_text):
        if not self.model:
            raise RuntimeError("Model not loaded or trained.")
        aspects = self.extract_aspects(review_text)
        results = []
        for aspect in aspects:
            preprocessed_review = self.preprocess_text(review_text + " " + aspect)
            prediction = self.model.predict([preprocessed_review])
            results.append((aspect, prediction[0]))
        return results
