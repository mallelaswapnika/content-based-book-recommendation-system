import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import re
from flask import Flask, render_template, request

# --- 1. SETUP AND MODEL TRAINING (Your Original Code) ---
try:
    # Load the books DataFrame
    books_df = pd.read_csv("1.csv", low_memory=False)
except FileNotFoundError:
    print("Error: '1.csv' not found. Please ensure the file is in the project directory.")
    # Exit or handle gracefully in a real application
    exit()

# Combine textual information to simulate a "description"
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)  # remove non-alphabetic characters
    return text

# Create a synthetic "description" from title, author, and publisher
books_df['combined_text'] = (
    books_df['Book-Title'].fillna('') + ' ' +
    books_df['Book-Author'].fillna('') + ' ' +
    books_df['Publisher'].fillna('')
)

books_df['processed_description'] = books_df['combined_text'].apply(preprocess_text)

# Vectorize using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(books_df['processed_description'])

# Train Nearest Neighbors model
n_neighbors = 6  # Number of neighbors to return, including the input book
model = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
model.fit(tfidf_matrix)

# Create a mapping from book titles to their index
# Note: Use a clean list of titles for autocompletion on the frontend if needed
title_to_index = pd.Series(books_df.index, index=books_df['Book-Title'].str.lower())

# Recommendation function
def recommend_books(title, model=model, books_df=books_df, title_to_index=title_to_index):
    title_lower = title.lower()
    if title_lower not in title_to_index:
        # Try to find a match that starts with the title
        matched_titles = [t for t in title_to_index.index if t.startswith(title_lower)]
        if not matched_titles:
            return None 

        # Use the first best match found
        idx = title_to_index[matched_titles[0]]
        print(f"Using best match: {books_df.loc[idx, 'Book-Title']}")

    else:
        idx = title_to_index[title_lower]
        
    distances, indices = model.kneighbors(tfidf_matrix[idx])
    indices = indices[0][1:]  # exclude the input book itself
    
    # Return a list of dictionaries with more details if available
    recommendations = books_df.iloc[indices][['Book-Title', 'Book-Author', 'Year-Of-Publication']].to_dict('records')
    return recommendations


# --- 2. FLASK APPLICATION ---

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = None
    input_title = None
    error_message = None
    
    if request.method == 'POST':
        # Get the book title from the web form
        input_title = request.form.get('book_title')
        
        if input_title:
            # Call your recommendation function
            recommendations = recommend_books(input_title)
            
            if recommendations is None:
                error_message = f"Sorry! Book '{input_title}' not found in the database. Try another book!"
        else:
             error_message = "Please enter a book title."

    # Render the HTML template, passing the results
    return render_template(
        'index.html', 
        recommendations=recommendations, 
        input_title=input_title,
        error=error_message
    )

if __name__ == '__main__':
    # You must have 'Flask' installed: pip install Flask
    app.run(debug=True)
