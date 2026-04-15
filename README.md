# NewsRadar: Smart News Finder
NewsRadar is a high-performance Streamlit dashboard designed to aggregate, index, and rank news articles dynamically. It bridges the gap between raw web scraping and intelligent information retrieval.

## Key Features
Live Web Scraping: Pulls real-time headlines from BBC, Hindustan Times, Times of India, Indian Express, and NDTV.
Semantic Search Engine: Uses a TF-IDF Vectorizer with bigrams and Cosine Similarity to find articles that actually match your intent, not just exact keywords.
Customizable Discovery: Adjust the "depth" of the search by choosing how many pages to scrape per section and how many results to display.
Modern UI/UX: A custom-styled dark mode interface featuring glassmorphism, pulse animations, and responsive metrics tiles.
Duplicate Detection: Automatically filters out repetitive headlines across different sections or sources.

## Tech Stack
Frontend: Streamlit
Scraping: BeautifulSoup4 & requests
Data Analysis: pandas
Machine Learning: scikit-learn (TfidfVectorizer, Cosine Similarity)
NLP: nltk (Stopword removal)

## Prerequisites
Before running the app, ensure you have Python installed and the necessary NLTK corpora through cmd:
pip install streamlit requests beautifulsoup4 pandas scikit-learn nltk

## How to Run
Clone this repository or save the code to a file named news_recommender_app.py.
Open your terminal in the file directory.
Run the Streamlit server:
streamlit run news_recommender_app.py
Fetch News: Use the sidebar to select your sources and click "Fetch Latest News."
Search: Enter a topic (e.g., "SpaceX launch" or "Global Economy") to see ranked matches.

## How it Works
Ingestion: The app visits multiple URLs per news source and extracts anchor tags with substantial text.
Preprocessing: Text is cleaned by removing punctuation, converting to lowercase, and stripping out "stop words" (like the, is, and) using NLTK.
Vectorization: The user’s query and the article titles are converted into a mathematical matrix (TF-IDF).
Ranking: The "Match Score" is calculated using the dot product of vectors, showing you how closely a headline relates to your search query.
