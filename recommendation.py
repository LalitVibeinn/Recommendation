from flask import Flask, request, jsonify
import pandas as pd
from surprise import SVD, Dataset, Reader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load data (ensure these CSV files are in the same folder)
interactions = pd.read_csv('interactions.csv')
posts = pd.read_csv('posts.csv')
users = pd.read_csv('users.csv')

# Initialize Flask app
app = Flask(__name__)

# -----------------------------
# Collaborative Filtering Model
# -----------------------------
reader = Reader(rating_scale=(0, 5))
data = Dataset.load_from_df(
    interactions[['userID', 'postID', 'engagement']], reader)
trainset = data.build_full_trainset()
svd = SVD()
svd.fit(trainset)

# -------------------------
# Content-Based Filtering
# -------------------------
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(posts['tags'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# -------------------------
# Recommendation Functions
# -------------------------


def get_similar_posts(post_id, cosine_sim=cosine_sim):
    try:
        idx = posts[posts['postID'] == post_id].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        post_indices = [i[0] for i in sim_scores[1:11]]  # Top 10 similar posts
        return posts.iloc[post_indices]
    except IndexError:
        return pd.DataFrame()


def hybrid_recommend(user_id, post_id, weight_cf=0.7, weight_cb=0.3):
    cf_score = svd.predict(user_id, post_id).est
    similar_posts = get_similar_posts(post_id)
    cb_score = 1 if post_id in similar_posts['postID'].values else 0
    return weight_cf * cf_score + weight_cb * cb_score


def recommend_for_user(user_id, top_n=1000):
    post_ids = posts['postID'].unique()
    recommendations = [
        (int(post_id), float(hybrid_recommend(user_id, post_id)))
        for post_id in post_ids
    ]
    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)
    return recommendations[:top_n]

# -------------------------
# Flask Routes
# -------------------------


@app.route('/')
def home():
    return "✅ Flask Recommendation API is running locally!"


@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.json
        user_id = int(data['userID'])
        recommendations = recommend_for_user(user_id)

        return jsonify({
            'user_id': user_id,
            'recommendations': recommendations
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


# -------------------------
# Run the App
# -------------------------
if __name__ == '__main__':
    app.run(debug=True)
