# from flask import Flask, request, jsonify
# import pandas as pd
# from surprise import SVD, Dataset, Reader
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import linear_kernel

# # Load data (ensure these CSV files are in the same folder)
# interactions = pd.read_csv('interactions.csv')
# posts = pd.read_csv('posts.csv')
# users = pd.read_csv('users.csv')

# # Initialize Flask app
# app = Flask(__name__)

# # -----------------------------
# # Collaborative Filtering Model
# # -----------------------------
# reader = Reader(rating_scale=(0, 5))
# data = Dataset.load_from_df(
#     interactions[['userID', 'postID', 'engagement']], reader)
# trainset = data.build_full_trainset()
# svd = SVD()
# svd.fit(trainset)

# # -------------------------
# # Content-Based Filtering
# # -------------------------
# tfidf = TfidfVectorizer(stop_words='english')
# tfidf_matrix = tfidf.fit_transform(posts['tags'])
# cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# # -------------------------
# # Recommendation Functions
# # -------------------------


# def get_similar_posts(post_id, cosine_sim=cosine_sim):
#     try:
#         idx = posts[posts['postID'] == post_id].index[0]
#         sim_scores = list(enumerate(cosine_sim[idx]))
#         sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
#         post_indices = [i[0] for i in sim_scores[1:11]]  # Top 10 similar posts
#         return posts.iloc[post_indices]
#     except IndexError:
#         return pd.DataFrame()


# def hybrid_recommend(user_id, post_id, weight_cf=0.7, weight_cb=0.3):
#     cf_score = svd.predict(user_id, post_id).est
#     similar_posts = get_similar_posts(post_id)
#     cb_score = 1 if post_id in similar_posts['postID'].values else 0
#     return weight_cf * cf_score + weight_cb * cb_score


# def recommend_for_user(user_id, top_n=1000):
#     post_ids = posts['postID'].unique()
#     recommendations = [
#         (int(post_id), float(hybrid_recommend(user_id, post_id)))
#         for post_id in post_ids
#     ]
#     recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)
#     return recommendations[:top_n]

# # -------------------------
# # Flask Routes
# # -------------------------


# @app.route('/')
# def home():
#     return "✅ Flask Recommendation API is running locally!"


# @app.route('/recommend', methods=['POST'])
# def recommend():
#     try:
#         data = request.json
#         user_id = int(data['userID'])
#         recommendations = recommend_for_user(user_id)

#         return jsonify({
#             'user_id': user_id,
#             'recommendations': recommendations
#         })

#     except Exception as e:
#         return jsonify({'error': str(e)}), 400


# # -------------------------
# # Run the App
# # -------------------------
# if __name__ == '__main__':
#     app.run(debug=True)


import psycopg2
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from surprise import SVD, Dataset, Reader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import MinMaxScaler

conn = psycopg2.connect(
    dbname="body_clone",
    user="body_clone_user",
    password="BlBiFsnCWhMDPm6xYWhuWeUn7EdfGrS8",
    host="dpg-cuknfh5ds78s739q4ca0-a.oregon-postgres.render.com",
    port="5432"
)

users_query = """
SELECT 
    "userId", 
    "username", 
    "interests", 
    "vibeScore",
    "vibeCount",
    "postCount",
    "followers_count",
    "dailyCommentCount"
FROM users
"""

posts_query = """
SELECT 
    id AS "postID", 
    caption AS "content"
FROM posts
"""

users_df = pd.read_sql(users_query, conn)
posts_df = pd.read_sql(posts_query, conn)

posts_df['sentiment_score'] = 0
posts_df['tags'] = None

users_df['userId'] = users_df['userId'].astype(str)
posts_df['postID'] = posts_df['postID'].astype(str)

interactions = []

for _, user in users_df.iterrows():
    viewed_posts = np.random.choice(
        posts_df['postID'], size=min(10, len(posts_df)), replace=False)
    for post_id in viewed_posts:
        base_score = (
            user['postCount'] +
            user['vibeScore'] +
            user['followers_count'] +
            user['dailyCommentCount']
        )
        noise = np.random.uniform(0.6, 1.2)
        score = base_score * noise
        interactions.append([user['userId'], post_id, score])

interactions = pd.DataFrame(interactions, columns=[
                            'userID', 'postID', 'engagement'])


scaler = MinMaxScaler(feature_range=(0, 5))
interactions['engagement'] = scaler.fit_transform(interactions[['engagement']])

reader = Reader(rating_scale=(0, 5))
data = Dataset.load_from_df(
    interactions[['userID', 'postID', 'engagement']], reader)
trainset = data.build_full_trainset()
svd = SVD()
svd.fit(trainset)


tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(posts_df['content'].fillna(''))
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


app = Flask(__name__)


def get_similar_posts(post_id, cosine_sim=cosine_sim):
    try:
        idx = posts_df[posts_df['postID'] == post_id].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        post_indices = [i[0] for i in sim_scores[1:11]]
        return posts_df.iloc[post_indices]
    except:
        return pd.DataFrame()


def hybrid_recommend(user_id, post_id, weight_cf=0.7, weight_cb=0.3):
    cf_score = svd.predict(user_id, post_id).est
    similar_posts = get_similar_posts(post_id)
    cb_score = 1 if post_id in similar_posts['postID'].values else 0
    return weight_cf * cf_score + weight_cb * cb_score


def recommend_for_user(user_id, top_n=1000):
    post_ids = posts_df['postID'].unique()
    recommendations = [
        (post_id, float(hybrid_recommend(user_id, post_id)))
        for post_id in post_ids
    ]
    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)
    return recommendations[:top_n]


@app.route('/')
def home():
    return "✅ Flask Recommendation API is running with UUIDs!"


@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.json
        user_id = str(data['userID'])
        recommendations = recommend_for_user(user_id)
        return jsonify({
            'user_id': user_id,
            'recommendations': recommendations
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
