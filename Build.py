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

DATA_PATH = "Data/pa1-data/"

# 1 - Import de la collection

# Like map(), but built to accept multiple functions


def map_many(iterable, *functions):
    if len(functions) == 0:
        return iterable
    if len(functions) == 1:
        return map(functions[0], iterable)
    return map_many(map(functions[0], iterable), *functions[1:])


def loadData(rootPath, shouldKeep, *processors):
    '''
    loadData loads the dataset at the given path. It takes a 'shouldKeep' filter
    function that can be used to filter out unwanted terms (e.g. stop words).
    Each word will then go through the processors (in the order as they are
    passed). Note that this function will return 'corpus' which is a slice of
    iterators: when the function returns, the filters and processors will not
    have been computed yet.
    '''

    print("Loading dataset")
    filenames = []
    corpus = {}
    i = 0

    for dirName in sorted(listdir(rootPath)):
        dirPath = join(rootPath, dirName)
        print(f"Loading files from {dirPath}...")

        for filename in listdir(dirPath):
            with open(join(dirPath, filename), 'r') as f:
                filenames.append(join(dirName, filename))
                # skipping tokenization as collection is already tokenized
                corpus[i] = map_many(
                    filter(shouldKeep, f.read().split(' ')), *processors)
                i += 1
    return corpus, filenames


# Stop word remover
stopWords = set(stopwords.words('english'))


def isNotStopWord(word): return word not in stopWords

# Convert to lowercase


def lowerize(word): return word.lower()

# We don't stem and go straight to lemmatization as it provides better results
# Stemmer = PorterStemmer()
# stem = lambda word: Stemmer.stem(word)


# Lemmatizer
# Using memoization (lru_cache) cache here gives a significant speed-up
Lemmatizer = WordNetLemmatizer()
lemmatize = lru_cache(maxsize=None)(Lemmatizer.lemmatize)

corpus, Filenames = loadData(DATA_PATH, isNotStopWord, lowerize, lemmatize)


# 2 - Calcul de l'index inversé

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
                inverted_index[term] = {document: 1}

    sys.stdout.write(f"Processing: {filecount} / {filecount}\r\n")
    sys.stdout.flush()
    return inverted_index


inverted_index = build_inverted_index(corpus)


# 3 - Sauvegarde de l'index inversé et des Filenames

def save_inverted_index_pickle(inverted_index, filename):
    print("Saving inverted index to disk...")
    with open(filename, "wb") as f:
        pickle.dump(inverted_index, f)
        f.close()


save_inverted_index_pickle(inverted_index, "inverted_index")

json = json.dumps(Filenames)
f = open("Filenames.json", "w")
f.write(json)
f.close()
