import os
import re
import fitz
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def find_pdfs_in_folder(folder_path):
    pdf_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".pdf"):
                full_path = os.path.join(root, file)
                label = 1 if 'publishable' in root.lower() and 'non' not in root.lower() else 0
                conf = None
                for c in ['cvpr', 'neurips', 'emnlp', 'tmlr', 'kdd']:
                    if c in root.lower():
                        conf = c
                pdf_files.append((file, full_path, label, conf))
    return pdf_files

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return ' '.join(page.get_text() for page in doc)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return ' '.join(text.split())

def tokenize_and_lemmatize(text):
    tokens = nltk.word_tokenize(text)
    return ' '.join(lemmatizer.lemmatize(word) for word in tokens if word not in stop_words)

def preprocess_documents(texts):
    return [tokenize_and_lemmatize(clean_text(t)) for t in texts]

reference_folder_path = "C:/Users/shash/Downloads/Reference"

pdfs = find_pdfs_in_folder(reference_folder_path)

texts = []
binary_labels = []
multiclass_labels = []

for _, path, pub_label, conf_label in pdfs:
    raw_text = extract_text_from_pdf(path)
    texts.append(raw_text)
    binary_labels.append(pub_label)
    multiclass_labels.append(conf_label if conf_label else "na")

processed_texts = preprocess_documents(texts)

vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(processed_texts)

X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(X, binary_labels, test_size=0.2, random_state=42)

conf_labels = [c for c in multiclass_labels if c != "na"]
conf_texts = [processed_texts[i] for i in range(len(multiclass_labels)) if multiclass_labels[i] != "na"]
X_conf = vectorizer.transform(conf_texts)

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_conf_encoded = label_encoder.fit_transform(conf_labels)

X_train_conf, X_test_conf, y_train_conf, y_test_conf = train_test_split(X_conf, y_conf_encoded, test_size=0.2, random_state=42)
