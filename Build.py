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
import sys

# Config

MAX_FILES_TO_INDEX = 200000
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

        print(f"Loading files from {dirPath}...")

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


def build_inverted_index(collection):
    '''builds the inverted index'''
    print("Parsing files and building inverted index")
    filecount = len(collection)
    current = 0
    inverted_index = {}
    for document in collection:

        # Print progress
        current += 1
        if current % 100 == 0:
            sys.stdout.write(f"Processing: {current} / {filecount}\r")
            sys.stdout.flush()
        
        for term in collection[document]:
            if term in inverted_index:
                if document in inverted_index[term]:
                    inverted_index[term][document] += 1
                else:
                    inverted_index[term][document] = 1
            else:
                inverted_index[term] = {}
                inverted_index[term][document] = 1

    sys.stdout.write(f"Processing: {filecount} / {filecount}\r\n")
    sys.stdout.flush()
    return inverted_index


inverted_index = build_inverted_index(corpus)

# 4 - Sauvegarder la matrice d'occurences et les Filenames.


def save_inverted_index_pickle(inverted_index, filename):
    print("Saving the index")
    with open(filename, "wb") as f:
        pickle.dump(inverted_index, f)
        f.close()


save_inverted_index_pickle(inverted_index, "inverted_index")
