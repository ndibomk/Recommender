from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import pickle

application = Flask(__name__)

# Load the processed data and similarity matrix from pickle files
movies = pickle.load(open('movie_list.pkl', 'rb'))
# similarity_matrix = pickle.load(open('similarity.pkl', 'rb'))

# Define recommend function based on processed data and similarity matrix
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movies['Combined_Text']).toarray()

# Convert dense matrix to sparse matrix
sparse_tfidf_matrix = csr_matrix(tfidf_matrix)

def recommend_using_pickles(input_text, num_recommendations=10):
    input_vector = tfidf_vectorizer.transform([input_text]).toarray()
    similarity_scores = cosine_similarity(input_vector, tfidf_matrix)
    similar_indices = similarity_scores.argsort()[0][::-1]  # Highest similarity first
    
    recommended_categories = []
    for idx in similar_indices:
        if idx != 0:  # Exclude the input category itself
            recommended_categories.append(movies.iloc[idx]['Category'])
            if len(recommended_categories) >= num_recommendations:
                break
    
    return recommended_categories

@application.route('/', methods=['POST'])
def recommend_endpoint():
    data = request.json
    input_text = data.get('input_text', '')
    num_recommendations = data.get('num_recommendations', 10)
    
    recommended_categories = recommend_using_pickles(input_text, num_recommendations)
    response = {'recommended_categories': recommended_categories}
    
    return jsonify(response)

if __name__ == '__main__':
    application.run(debug=True)
