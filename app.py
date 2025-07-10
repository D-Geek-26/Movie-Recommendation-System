from flask import Flask,request,render_template
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
app=Flask(__name__)

# Loading the dataset
df=pd.read_csv('movie.csv')

# Creating the genre matrix
def create_genre_matrix(df):
    genres=df['genres'].str.split('|')
    unique_genres=sorted(set(g for genre_list in genres for g in genre_list))
    genre_matrix=pd.DataFrame(0,index=df.index,columns=unique_genres)
    for idx,genre_list in enumerate(genres):
        for genre in genre_list:
            genre_matrix.at[idx,genre]=1
    return genre_matrix,unique_genres

genre_matrix,unique_genres=create_genre_matrix(df)

# Creating the user genre vector
def get_user_genre_vector(user_genres,unique_genres):
    user_vector = np.zeros(len(unique_genres))
    for genre in user_genres:
        if genre in unique_genres:
            user_vector[unique_genres.index(genre)]=1
    return user_vector.reshape(1,-1)

# Home route to render the form
@app.route('/',methods=['GET'])
def home():
    return render_template('Movie.html',recommendations=None,error=None,genres=unique_genres)

# Recommendation endpoint
@app.route('/recommend',methods=['POST'])
def recommend():
    genres_input=request.form.get('genres', '').strip()
    
    if not genres_input:
        return render_template('Movie.html',recommendations=None,error="Please enter at least one genre.",genres=unique_genres)
    
    # Parse user genres
    user_genres=[g.strip() for g in genres_input.split(',')]
    invalid_genres=[g for g in user_genres if g not in unique_genres]
    
    if invalid_genres:
        return render_template('Movie.html',recommendations=None,error=f"Invalid genres: {', '.join(invalid_genres)}. Valid genres are: {', '.join(unique_genres)}",genres=unique_genres)
    
    # Create user genre vector
    user_vector=get_user_genre_vector(user_genres, unique_genres)
    
    # Compute cosine similarity
    similarities=cosine_similarity(user_vector, genre_matrix)[0]
    
    # Get top 5 movie indices
    top_indices=np.argsort(similarities)[::-1][:5]
    
    # Prepare recommendations
    recommendations=[]
    for idx in top_indices:
        recommendations.append({
            'title': df.iloc[idx]['title'],
            'genres': df.iloc[idx]['genres'],
            'similarity': similarities[idx]
        })
    
    return render_template('Movie.html',recommendations=recommendations,error=None,genres=unique_genres)

if __name__ =='__main__':
    app.run(debug=True)