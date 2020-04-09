import pickle


def load_inverted_index_pickle(filename):
    with open(filename, 'rb') as fb:
        index = pickle.load(fb)
        return index


index = load_inverted_index_pickle("dump")

# 4 - Parser les requetes


def loadQueries():
    Queries = []
    i = 0
    for i in range(1, 9):
        with open("Queries/dev_queries/query."+str(i), 'r') as f:
            Queries.append(f.read())
    return Queries


Queries = loadQueries()

print(Queries)

# Cf exemples dans Queries/ -> stanford students

# 5 - Executer des requetes

# Cf Utils/Lab2.py

# 6 - Afficher Résultats

# Cf exemples dans Queries/ -> Liste ordonnée de n (?) documents

# 7 - Evaluer résultats

# Cf Queries/dev_output
