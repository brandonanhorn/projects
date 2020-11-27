import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

df = pd.read_csv("data/books.csv")

df.head(1

avg_rating_df = df[["title", "average_rating"]]
avg_rating_df.head(1)

def create_csr_X_matrix(df):
    m_len = len(avg_rating_df.title.unique())
    book_encoder = LabelEncoder()
    m = book_encoder.fit_transform(avg_rating_df.title)
    X = csr_matrix((avg_rating_df["average_rating"]))
    return X, book_encoder

X, book_encoder = create_csr_X_matrix(avg_rating_df)

book = "Gilead"

#def find_me_a_book(book_title, X, k=10):
book_ind = title_encoder.transform([book])
#error here
book_vec = X[book_ind]
kNN = NearestNeighbors(n_neighbors=k, algorithm="brute", metric='minkowski')
kNN.fit(X)
neighbour = kNN.kneighbors(book_vec, return_distance=False)
close_movies = neighbour.flatten().tolist()
z = [book_encoder.inverse_transform([m])[0] for m in close_books]
z = [zi for zi in z if zi != book_title]
    return z

find_me_a_book(book, X, k=10)
