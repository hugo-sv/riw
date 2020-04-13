import json
import pickle
from os import listdir
from os.path import isfile, join
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from collections import OrderedDict
from functools import lru_cache

# Config

MAX_FILES_TO_INDEX = 7000
DATA_PATH = "Data/pa1-data/"

# 1 - Import et processing de la collection

# Like map(), but built to accept multiple functions
def map_many(iterable, *functions):
    if len(functions) == 0:
        return iterable
    if len(functions) == 1:
        return map(functions[0], iterable)
    return map_many(map(functions[0], iterable), *functions[1:])


def loadData(rootPath, shouldKeep, *processors):
    '''loadData Loads the dataset at the given path.
    It must be in the following format : '''
    print("Loading the dataset")
    Filenames = []
    corpus = {}
    indexed = 0

    for dirName in sorted(listdir(rootPath)):
        dirPath = join(rootPath, dirName)

        # skip files
        if isfile(dirPath):
            continue

        print(f"Parsing directory {dirPath}...")

        for filename in listdir(dirPath):
            filePath = join(dirPath, filename)

            if isfile(filePath):
                with open(filePath, 'r') as f:
                    # skipping tokenization as collection is already tokenized
                    corpus[indexed] = map_many(filter(shouldKeep, f.read().split(' ')), *processors)
                    Filenames.append(join(dirName, filename))
                    indexed += 1
                    if indexed > MAX_FILES_TO_INDEX:
                        return corpus, Filenames
    return corpus, Filenames

# Stop word remover
stopWords = set(stopwords.words('english'))
isNotStopWord = lambda word: word not in stopWords

# Convert to lowercase
lowerize = lambda word: word.lower()

# We don't stem and go straight to lemmatization as it provides better results
# Stemmer = PorterStemmer()
# stem = lambda word: Stemmer.stem(word)

# Lemmatizer
# Using memoization (lru_cache) cache here gives a ~x4 speed-up
Lemmatizer = WordNetLemmatizer()
lemmatize = lru_cache(maxsize=None)(Lemmatizer.lemmatize)

corpus, Filenames = loadData(DATA_PATH, isNotStopWord, lowerize, lemmatize)

json = json.dumps(Filenames)
f = open("Filenames.json", "w")
f.write(json)
f.close()


# 3 - Calculer la matrice d'occurences


def build_inverted_index(collection, type_index):
    '''builds the inverted index the requested type_index'''
    print("Building inverted index")
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


inverted_index = build_inverted_index(corpus, 1)

# 4 - Sauvegarder la matrice d'occurences et les Filenames.


def save_inverted_index_pickle(inverted_index, filename):
    print("Saving the index")
    with open(filename, "wb") as f:
        pickle.dump(inverted_index, f)
        f.close()


save_inverted_index_pickle(inverted_index, "inverted_index")
