import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

df = pd.read_csv("data/books.csv")
df = df.fillna(0)

df.head(1)

def create_csr_X_matrix(df):
    book_title_len = len(df.title.unique())
    unique_book_len = len(df.isbn13.unique())

    title_encoder = LabelEncoder()
    unique_id_encoder = LabelEncoder()

    t = title_encoder.fit_transform(df.title)
    b = unique_id_encoder.fit_transform(df.isbn13)

    book_title_len = len(df.title.unique())
    unique_book_len = len(df.isbn13.unique())

    X = csr_matrix((df["average_rating"], (t, b)), shape=(book_title_len, unique_book_len))
    return X, title_encoder, unique_id_encoder


X, title_encoder, unique_id_encoder = create_csr_X_matrix(df)

book = 'Gilead'

def find_me_a_book_to_read(book_title, X, k=10):
    book_ind = movie_encoder.transform([book])
    book_vec = X[book_ind]
    kNN = NearestNeighbors(n_neighbors=k, algorithm="brute", metric='minkowski')
    kNN.fit(X)
    neighbour = kNN.kneighbors(book_vec, return_distance=False)
    close_books = neighbour.flatten().tolist()
    z = [title_encoder.inverse_transform([t])[0] for t in close_books]
    z = [zi for zi in z if zi != book_title]
    return z

find_me_a_book_to_read("Hour Game", X, k=10)
