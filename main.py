import requests
import re
import spacy
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

# Adresy URL książek jako słownik z tytułami jako klucze
book_urls = {
    "Pan Tadeusz": "https://wolnelektury.pl/media/book/txt/pan-tadeusz.txt",
    "Lalka Tom Pierwszy": "https://wolnelektury.pl/media/book/txt/lalka-tom-pierwszy.txt",
    "Calineczka": "https://wolnelektury.pl/media/book/txt/calineczka.txt",
    "Zemsta": "https://wolnelektury.pl/media/book/txt/zemsta.txt",
    "Syzyfowe prace": "https://wolnelektury.pl/media/book/txt/syzyfowe-prace.txt"
}

# Pobieranie tekstu z URL
def get_text(url):
    response = requests.get(url)
    if response.status_code == 200:
        text = response.text
        text_without_digits = re.sub(r'\d', '', text)
        return text_without_digits
    else:
        return ""

# Pobranie i przetworzenie tekstów
corpus = [get_text(url) for url in book_urls.values()]

# Pobranie listy polskich stopwords z GitHuba
url_stopwords = 'https://raw.githubusercontent.com/bieli/stopwords/master/polish.stopwords.txt'
response = requests.get(url_stopwords)
custom_stopwords = set(response.text.splitlines())

# Załadowanie polskiego modelu
nlp = spacy.load('pl_core_news_md')

# Tokenizacja, usuwanie stopwords i lematyzacja
processed_corpus = []
for text in corpus:
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if token.lemma_.lower() not in custom_stopwords and token.is_alpha]
    processed_corpus.append(' '.join(tokens))

# Sprawdzenie, czy przetworzony korpus zawiera dane
for doc in processed_corpus:
    if len(doc.strip()) == 0:
        print("Uwaga: Jeden z dokumentów jest pusty po przetworzeniu.")
    else:
        print("Fragment dokumentu:", doc[:100])  # Wyświetlenie fragmentu dokumentu

# TF-IDF
vectorizer = TfidfVectorizer()
try:
    tfidf_matrix = vectorizer.fit_transform(processed_corpus)
except ValueError as e:
    print("Błąd przy tworzeniu macierzy TF-IDF:", e)
    exit()

# Podobieństwo cosinusowe
cosine_sim_matrix = cosine_similarity(tfidf_matrix)

# Redukcja wymiarowości i wizualizacja
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(tfidf_matrix.toarray())

# Tytuły książek
book_titles = list(book_urls.keys())

# Rysowanie wykresu
plt.figure(figsize=(10, 8))
for i, (x, y) in enumerate(reduced_data):
    plt.scatter(x, y, label=book_titles[i])
plt.legend()
plt.title('Redukcja wymiarowości macierzy TF-IDF')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()

def plot_token_counts(processed_corpus, book_titles):

    # zliczenie numerów w książce
    token_counts = [len(text.split()) for text in processed_corpus]

    # słupki!
    plt.figure(figsize=(12, 6))
    plt.bar(book_titles, token_counts, color='teal')
    plt.xlabel('Books')
    plt.ylabel('Number of Tokens')
    plt.title('Token Counts in Each Book After Lemmatization')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Wypisywanie tokenów i ich liczby
for title, text in zip(book_titles, processed_corpus):
    token_set = set(text.split())
    print(f"Book: {title}")
    print(f"Number of Unique Tokens: {len(token_set)}")
    print(f"Sample Tokens: {list(token_set)[:10]}")
    print()

# Wywołanie funkcji do stworzenia wykresu liczby tokenów
plot_token_counts(processed_corpus, book_titles)
