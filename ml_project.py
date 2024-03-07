from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('stopwords')
nltk.download('punkt')

preprocessed_texts_lst = preprocess_text(received_texts)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(preprocessed_texts_lst)

num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(X)
clusters = kmeans.labels_.tolist()

food_poisoning_cluster = identify_food_poisoning_cluster(clusters, preprocessed_texts_lst)

project_brief = generate_project_brief(food_poisoning_cluster, additional_inputs)

def preprocess_text(texts):
    preprocessed_texts_lst = []
    stop_words = set(stopwords.words('english'))
    for text in texts:
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
        preprocessed_text = ' '.join(tokens)
        preprocessed_texts_lst.append(preprocessed_text)
    
    return preprocessed_texts_lst


def identify_food_poisoning_cluster(clusters, texts):
    food_poisoning_keywords = ['food poisoning', 'nausea', 'vomiting', 'diarrhea', 'stomach ache']
    
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)

    food_poisoning_similarities = []
    for keyword in food_poisoning_keywords:
        keyword_vector = vectorizer.transform([keyword])
        similarities = cosine_similarity(X, keyword_vector)
        food_poisoning_similarities.append(similarities.flatten())

    avg_similarities = np.mean(food_poisoning_similarities, axis=0)

    food_poisoning_cluster = [[] for _ in range(max(clusters) + 1)]
    for i, cluster_label in enumerate(clusters):
        food_poisoning_cluster[cluster_label].append((texts[i], avg_similarities[i]))

    for i in range(len(food_poisoning_cluster)):
        food_poisoning_cluster[i] = sorted(food_poisoning_cluster[i], key=lambda x: x[1], reverse=True)

    return food_poisoning_cluster


def generate_project_brief(food_poisoning_cluster, additional_inputs):
    total_texts = sum(len(cluster) for cluster in food_poisoning_cluster)
    food_poisoning_texts = sum(len(cluster) for cluster in food_poisoning_cluster)
    other_texts = total_texts - food_poisoning_texts

    accuracy = food_poisoning_texts / total_texts if total_texts > 0 else 0.0

    project_brief = f"Project Brief:\n"
    project_brief += f"Total number of texts: {total_texts}\n"
    project_brief += f"Number of food poisoning cases: {food_poisoning_texts}\n"
    project_brief += f"Number of other texts: {other_texts}\n"
    project_brief += f"Accuracy of classification: {accuracy:.2f}\n"

    return project_brief
