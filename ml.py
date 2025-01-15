
import solara
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

music_df = pd.read_csv(r"c:\Users\Arogya Mary\Downloads\dataset.csv\dataset.csv")
music_df = music_df.dropna()

feature_columns = [
    'danceability', 'energy', 'key', 'loudness', 'mode',
    'speechiness', 'acousticness', 'instrumentalness', 'liveness',
    'valence', 'tempo'
]

scaler = MinMaxScaler()
music_features = music_df[feature_columns].values
music_features_scaled = scaler.fit_transform(music_features)

def calculate_weighted_popularity(popularity):
    return popularity / 100

def content_based_recommendations(input_song, num_recommendations):
    if input_song not in music_df['track_name'].values:
        return None
    input_song_index = music_df[music_df['track_name'] == input_song].index[0]
    similarity_scores = cosine_similarity([music_features_scaled[input_song_index]], music_features_scaled)
    similar_song_indices = similarity_scores.argsort()[0][::-1][1:num_recommendations + 1]
    return music_df.iloc[similar_song_indices][['track_name', 'artists', 'album_name', 'popularity']]

def hybrid_recommendations(input_song, num_recommendations):
    content_rec = content_based_recommendations(input_song, num_recommendations)
    if content_rec is None:
        return None
    popularity_score = music_df.loc[music_df['track_name'] == input_song, 'popularity'].values[0]
    weighted_popularity_score = calculate_weighted_popularity(popularity_score)
    input_song_data = {
        'track_name': [input_song],
        'artists': [music_df.loc[music_df['track_name'] == input_song, 'artists'].values[0]],
        'album_name': [music_df.loc[music_df['track_name'] == input_song, 'album_name'].values[0]],
        'popularity': [weighted_popularity_score]
    }
    input_song_df = pd.DataFrame(input_song_data)
    hybrid_rec = pd.concat([content_rec, input_song_df], ignore_index=True)
    hybrid_rec = hybrid_rec.sort_values(by='popularity', ascending=False)
    return hybrid_rec[hybrid_rec['track_name'] != input_song]

@solara.component
def MusicRecommendationApp():

    input_song, set_input_song = solara.use_state("")
    num_recommendations, set_num_recommendations = solara.use_state(5)

    # Input field for the song name
    solara.InputText( label="Enter a song name",  value=input_song.title(),  on_value=lambda value: set_input_song(value)  )
    submit_button = solara.Button(label="Get Recommendations")

    with solara.Column():
        solara.Markdown("# 🎵 Music Recommendation System")
        solara.Markdown("Enter a song name and get similar recommendations!")
        if submit_button:
            if input_song:
                recommendations = hybrid_recommendations(str(input_song), num_recommendations)
                if recommendations is not None:
                    solara.Markdown(f"## Recommendations for '{input_song}'")
                    solara.DataFrame(recommendations)
                else:
                    solara.Markdown(f"No recommendations found for '{input_song}'. Try another song.")



# Run the app
if __name__ == "__main__":
     MusicRecommendationApp()