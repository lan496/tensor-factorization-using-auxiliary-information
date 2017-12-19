import numpy as np
import pandas as pd


def load():
    users_cols = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
    users = pd.read_csv('ml-100k/u.user', sep='|', names=users_cols)

    occupations_lst = np.loadtxt('ml-100k/u.occupation', dtype=str)

    genres_cols = ['genre', 'genre_id']
    genres = pd.read_csv('ml-100k/u.genre', sep='|', names=genres_cols)

    genres_lst = genres.sort_values(by='genre_id')['genre'].values

    movies_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL', *genres_lst]
    movies = pd.read_csv('ml-100k/u.item', sep='|', names=movies_cols, encoding='latin-1')

    data_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
    data = pd.read_csv('ml-100k/u.data', sep='\t', names=data_cols, encoding='latin-1')

    sim_users = users_similarity(usrs)
    sim_movies = movies_similarity(movies)


def users_similarity(users):
    users_tmp = pd.get_dummies(users, columns=['occupation', 'gender'], drop_first=True)
    users_tmp.drop(['zip_code', 'user_id'], axis=1, inplace=True)
    scaler = MinMaxScaler()
    users_tmp['age'] = scaler.fit_transform(users_tmp['age'].astype(float).values.reshape(-1, 1))
    
    X = users_tmp.values
    
    n = X.shape[0]
    sim = np.zeros((n,  n))
    for i in range(n):
        for j in range(n):
            sim[i, j] = np.dot(X[i, :], X[j, :]) / np.linalg.norm(X[i, :]) / np.linalg.norm(X[j, :])

    return sim
