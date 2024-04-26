import xml.etree.ElementTree as ET
import re
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

# Ensure NLTK resources are downloaded
# nltk.download('punkt')
# nltk.download('wordnet')

class Predicting():
    def __init__(self):
        # Initialize your class attributes here if needed
        pass

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = word_tokenize(text)
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
        return ' '.join(lemmatized_tokens)

    def parse_xml_to_dataset(self, xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        texts = []
        labels = []
        for review in root.findall('Review'):
            sentences = review.find('sentences')
            for sentence in sentences.findall('sentence'):
                text = sentence.find('text').text
                opinions = sentence.find('Opinions')
                if opinions is not None:
                    for opinion in opinions.findall('Opinion'):
                        preprocessed_text = self.preprocess_text(text)
                        texts.append(preprocessed_text)
                        labels.append(opinion.get('polarity'))
        return texts, labels

    def train_and_save_model(self, xml_file):
        texts, labels = self.parse_xml_to_dataset(xml_file)
        X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

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

        predictions = grid_search.best_estimator_.predict(X_test)
        print("Classification Report:\n", classification_report(y_test, predictions))

    def predict_opinion(self, review_text):
        with open('svm_pipeline.pkl', 'rb') as model_file:
            trained_model = pickle.load(model_file)
        preprocessed_review = self.preprocess_text(review_text)
        prediction = trained_model.predict([preprocessed_review])
        return prediction[0]


