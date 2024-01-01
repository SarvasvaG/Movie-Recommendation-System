import streamlit as st 
import pickle
import numpy as np
import pandas as pd
import requests

with open('Models/user_based.pcl', 'rb') as f:
    model = pickle.load(f)

with open('Models/movie_based.pcl', 'rb') as f:
    model_m = pickle.load(f)

with open('Models/movie_unique_data.pcl', 'rb') as f:
    movies_unique_data = pickle.load(f)

with open('Models/movies_vecs.pcl', 'rb') as f:
    movies_vecs = pickle.load(f)
    
with open('Models/scaler_users.pcl', 'rb') as f:
    scaler_users = pickle.load(f)

with open('Models/scaler_movies.pcl', 'rb') as f:
    scaler_movies = pickle.load(f)

with open('Models/scaler_ratings.pcl', 'rb') as f:
    scaler_ratings = pickle.load(f)

n = len(movies_vecs)
movie_id_to_tmdb = pd.read_csv('ml-latest-small/links.csv')
movie_id_to_tmdb.set_index('movieId', inplace=True)
movie_id_to_tmdb = movie_id_to_tmdb.drop(['imdbId'], axis='columns')['tmdbId']
vms = model_m.predict(scaler_movies.transform(movies_vecs))
number_of_movies = vms.shape[0]
    
genres = {
    'Action':0,	'Adventure':0,	'Animation':0,	'Children':0,	'Comedy':0,	'Crime':0,	'Non-Fiction':0,	'Drama':0,	'Fantasy':0,	'Horror':0,	'Musical':0,	'Mystery':0,	'Romance':0, 'Sci-Fi':0,	'Thriller':0
}

def sq_dist(a,b):
    return ((a-b)**2).sum()

def calculate_distance(movie_index):
    distance = np.zeros((1, number_of_movies))

    for i in range(number_of_movies):
        if movie_index != i:
            distance[0,i]=sq_dist(vms[movie_index], vms[i])
        else:
            distance[0,movie_index]=np.inf
    return distance     


def get_posters(closest_eight_movies_ids):
    posters = []
    for id in closest_eight_movies_ids:
        response = requests.get(f'https://api.themoviedb.org/3/movie/{id}?api_key=31149e5d7ea8a663b9d1d120fd52033c')
        
        if response.status_code==200:
            data=response.json()
            poster_path = data.get('poster_path')
            
            if poster_path:
                posters.append(f'http://image.tmdb.org/t/p/w185/{data["poster_path"]}')
            else:
                posters.append('https://placehold.co/185x277/gray/white?text=Poster+Not+Found')
        else:
            posters.append('https://placehold.co/185x277/gray/white?text=Poster+Not+Found')
            print(f"Error: {response.status_code}")
            
    print(len(posters))
    return posters


def recommend_movies(movie_name,x):
    movie_index = movies_unique_data[movies_unique_data['title'] == movie_name].index[0]
    
    closest_eight_movies_indices = np.argsort(calculate_distance(movie_index).flatten())[x:x+8]
    closest_eight_movies_movie_ids = movies_unique_data['movieId'].loc[closest_eight_movies_indices]
    closest_eight_movies_ids = movie_id_to_tmdb[closest_eight_movies_movie_ids]
    
    closest_eight_movies = movies_unique_data['title'].iloc[closest_eight_movies_indices].to_list()
    closest_eight_movies_posters = get_posters(closest_eight_movies_ids)
    
    return closest_eight_movies, closest_eight_movies_posters

def recommend_movies_genres(genres,x):
    genres = {('Documentary' if k == 'Non-Fiction' else k): v for k, v in genres.items()}
    print(genres)
    
    user_pred_data = pd.Series(genres)
    users_vecs = pd.DataFrame([user_pred_data]*n, columns=user_pred_data.index)
    
    scaled_movies_vecs = scaler_movies.transform(movies_vecs)
    scaled_users_vecs = scaler_users.transform(users_vecs)
    
    scaled_rating_pred = model.predict([scaled_users_vecs, scaled_movies_vecs])
    rating_pred = scaler_ratings.inverse_transform(scaled_rating_pred)
    
    sorted_index = np.argsort(-rating_pred, axis=0).reshape(-1).tolist()
    
    sorted_movies_for_user = movies_unique_data.loc[sorted_index,:]
    temp = sorted_movies_for_user.iloc[x:x+8,:]
    
    closest_eight_movies_indices = temp.index
    closest_eight_movies_movie_ids = movies_unique_data['movieId'].loc[closest_eight_movies_indices]
    closest_eight_movies_ids = movie_id_to_tmdb[closest_eight_movies_movie_ids]
    
    closest_eight_movies = movies_unique_data['title'].iloc[closest_eight_movies_indices].to_list()
    closest_eight_movies_posters = get_posters(closest_eight_movies_ids)
    
    return closest_eight_movies, closest_eight_movies_posters

def display(names, posters):
    col1, col2, col3, col4 = st.columns(4)

    for i in range(8):
        with locals()[f'col{i % 4 + 1}']:
            st.text(names[i])
            st.image(posters[i])
 

session_state = st.session_state
if not hasattr(session_state, 'x'):
    session_state.x = 0
    
st.title('Movie Recommendation System')


st.header('Recommend Similar Movies')
option = st.selectbox('Search A Movie',movies_unique_data['title'].values)

if st.button('Get Movie Recommendations / Show More', key=1):
    names, posters = recommend_movies(option, session_state.x%40)
    
    st.subheader('Recommended Movies:')
    display(names, posters)
    
    session_state.x+=8
    
st.header('Suggest Movies Based on Genres')

colg1, colg2, colg3, colg4, colg5 = st.columns(5)
for i,genre in enumerate(genres):
    with locals()[f'colg{i % 5 + 1}']:
        genres[genre] = float(st.checkbox(genre, value=False))*5
        
if st.button('Get Movie Recommendations / Show More', key=2):
    names, posters = recommend_movies_genres(genres,session_state.x%24)
    
    st.subheader('Recommended Movies:')
    display(names, posters)
    
    session_state.x+=8
    
    



