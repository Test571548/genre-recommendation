import streamlit as st
import pandas as pd
import msgpack
import requests
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from collections import defaultdict



# Load the data
df = pd.read_csv("deezer_track_metadata_with_genre.csv")
# Define activity -> genre list mapping
activity_to_genres = {
    'studying': ['Classical', 'Jazz', 'Electro', 'Asian Music'],
    'driving': ['Pop', 'Rock', 'Dance', 'Electro', 'Rap/Hip Hop', 'Reggae'],
    'unwind':  ['Jazz', 'Classical', 'Reggae', 'Asian Music', 'Electro'],
    'party':   ['Dance', 'Pop', 'Rap/Hip Hop', 'Electro']
}

# Reverse mapping: genre -> list of activities
genre_to_activities = defaultdict(list)
for activity, genres in activity_to_genres.items():
    for genre in genres:
        genre_to_activities[genre].append(activity)

# Function to assign activity (choose first match, or join all)
def get_activity(genre):
    activities = genre_to_activities.get(genre, [])
    return ', '.join(activities) if activities else None  

# Apply the function to create the new column
df['activity'] = df['genre'].apply(get_activity)

print(df.head())

# Streamlit UI components
st.title("Music Track Metadata Analysis")

# Display the DataFrame
if st.checkbox("Show Data", value=True):
    st.write(df.head())

# Filter by genre
selected_genre = st.selectbox("Select Genre", df['genre'].unique())
filtered_df = df[df['genre'] == selected_genre]
st.write(f"Tracks in {selected_genre} genre", filtered_df)

# Display a bar chart of track durations by genre
st.subheader("Track Durations by Genre")
genre_duration = df.groupby('genre')['duration'].mean()
st.bar_chart(genre_duration)

# Train a KNN model to predict genre based on activity
activity_encoder = LabelEncoder()
genre_encoder = LabelEncoder()

df['activity_encoded'] = activity_encoder.fit_transform(df['activity'].str.split(',').str[0])
df['genre_encoded'] = genre_encoder.fit_transform(df['genre'])

X = df[['activity_encoded']]
y = df['genre_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

# Predict genre for a new activity
new_activity = st.text_input("Enter an activity to predict genre (e.g., 'studying')")
if new_activity:
    activity_encoded = activity_encoder.transform([new_activity])[0]
    predicted_genre_encoded = knn_model.predict([[activity_encoded]])[0]
    predicted_genre = genre_encoder.inverse_transform([predicted_genre_encoded])[0]
    st.write(f"Predicted Genre: {predicted_genre}")
