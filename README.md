# content-based-book-recommendation-system
Book Recommender System (Content-Based)
This is a full-stack content-based book recommendation engine built using Python, the Scikit-learn library for machine learning, and Flask to provide an interactive web interface.

The system recommends books based on the similarity of an input book's title, author, and publisher data, transforming these textual features into a numerical vector using TF-IDF.
Features:
1)Content-Based Filtering: Recommends similar books by analyzing the textual features of the books.
2)TF-IDF Vectorization: Converts book metadata into numerical vectors, emphasizing unique and important words.
3)Cosine Similarity & Nearest Neighbors: Uses the NearestNeighbors model with a cosine metric to find the books most similar in vector space.
4)Web Interface (Flask): A simple, user-friendly frontend allows users to input a book title and instantly view the top 5 recommendations.

Technologies Used:

Backend/ML	Python 3.x	Core programming language
Pandas	Data manipulation and processing
NumPy	Numerical operations
Scikit-learn	TF-IDF Vectorizer and Nearest Neighbors Model
Web Framework	Flask	Serves the web application and handles routing
Frontend	HTML, CSS, Jinja	User interface and template rendering
