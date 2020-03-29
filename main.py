import pickle
from os import listdir
from os.path import isfile, join
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
# 1 - Importer la collection

'''
Collection qui contient un ensemble de pages web du domaine stanford.edu. Les données
sont disponibles sur le Drive. C’est un corpus de 170 MBs. Il est organisé en 10 sous-répertoires (numérotés de 0 à 9). Chaque fichier correspond au contenu textuel d’une page web individuelle.
Chaque nom de fichier est unique dans chaque sous-répertoire mais ce n’est pas le cas globalement.
Cette collection est déjà tokenizée.
'''

# loadData Load the dataset


def loadData(dataPath):
    corpus = {}
    Filenames = []
    i = 0
    # In each directories
    for dir in [str(i) for i in range(10)]:
        directoryPath = dataPath+dir
        print("Parsing file "+dir + "/9...")
        # For each file
        for filename in listdir(directoryPath):
            filePath = join(directoryPath, filename)
            # If this is a file
            if isfile(filePath):
                with open(filePath, 'r') as f:
                    # Keeping dir as each filename is not necessarily unique
                    Filenames.append(join(dir, filename))
                    # Appending corpus, using Filenames' index
                    # Tokenizing with nltk here to save computation time
                    corpus[i] = word_tokenize(f.read())
                    i += 1
                    # Breaking point for testing purpose, @TODO Remove this
                    if i == 5:
                        return corpus, Filenames
    return corpus, Filenames


dataPath = "Data/pa1-data/"
corpus, Filenames = loadData(dataPath)
print(corpus, Filenames)

# 2 - Preprocess la collection

# remove_stop_words Remove stop words (from Lab1.py)


def remove_stop_words(collection, stop_word_file):
    collection_filtered = {}
    for i in collection:
        collection_filtered[i] = []
        for j in collection[i]:
            if j not in stop_word_file:
                collection_filtered[i].append(j)
    return collection_filtered


# Stemming (from Lab1.py)


def collection_stemming(segmented_collection):
    stemmed_collection = {}
    stemmer = PorterStemmer()  # initialisation d'un stemmer
    for i in segmented_collection:
        stemmed_collection[i] = []
        for j in segmented_collection[i]:
            stemmed_collection[i].append(stemmer.stem(j))
    return stemmed_collection


# Lemmatization (from Lab1.py)


def collection_lemmatize(segmented_collection):
    lemmatized_collection = {}
    stemmer = WordNetLemmatizer()  # initialisation d'un lemmatiseur
    for i in segmented_collection:
        lemmatized_collection[i] = []
        for j in segmented_collection[i]:
            lemmatized_collection[i].append(stemmer.lemmatize(j))
    return lemmatized_collection

# Apply stemming, or lemmatization ?

# 3 - Calculer la matrice d'occurences

# build_inverted_index builds the inverted index the requested type_index. (from Lab1.py)


def build_inverted_index(collection, type_index):
    # On considère ici que la collection est pré-traitée
    inverted_index = OrderedDict()
    if type_index == 1:
        for document in collection:
            for term in collection[document]:
                if term in inverted_index.keys():
                    if document not in inverted_index[term]:
                        inverted_index[term].append(document)
                else:
                    inverted_index[term] = [document]
    elif type_index == 2:
        for document in collection:
            for term in collection[document]:
                if term in inverted_index.keys():
                    if document in inverted_index[term].keys():
                        inverted_index[term][document] = inverted_index[term][document] + 1
                    else:
                        inverted_index[term][document] = 1
                else:
                    inverted_index[term] = OrderedDict()
                    inverted_index[term][document] = 1
    elif type_index == 3:
        for document in collection:
            n = 0
            for term in collection[document]:
                n = n+1
                if term in inverted_index.keys():
                    if document in inverted_index[term].keys():
                        inverted_index[term][document][0] = inverted_index[term][document][0] + 1
                        inverted_index[term][document][1].append(n)
                    else:
                        inverted_index[term][document] = [1, [n]]
                else:
                    inverted_index[term] = OrderedDict()
                    inverted_index[term][document] = [1, [n]]

    return inverted_index

# 4 - Sauvegarder la matrice d'occurences et les Filenames.

# Save and load, (from saveandload_pickle.py)


def save_inverted_index_pickle(inverted_index, filename):
    with open(filename, "wb") as f:
        pickle.dump(inverted_index, f)
        f.close()


def load_inverted_index_pickle(filename):
    with open(filename, 'rb') as fb:
        index = pickle.load(fb)
        return index

# @TODO Save Filenames

# 4 - Parser les requetes

# Cf exemples dans Queries/ -> stanford students

# 5 - Executer des requetes

# Cf Utils/Lab2.py

# 6 - Afficher Résultats

# Cf exemples dans Queries/ -> Liste ordonnée de n (?) documents

# 7 - Evaluer résultats

# Cf Queries/dev_output
