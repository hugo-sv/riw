import json
import pickle
from sys import stdout, version_info
from os import listdir
from os.path import isfile, join
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from collections import OrderedDict
from functools import lru_cache

# Config

DATA_PATH = "Data/pa1-data/"

# 0 - Check Python version

if version_info[0] < 3 or version_info[1] < 7:
    print("This script *requires* Python 3.7+. See README")
    print(f"(you are running Python {'.'.join(map(str, version_info[:3]))})")
    exit(1)


# 1 - Import the dataset

def map_many(iterable, *functions):
    '''Like map(), but built to accept multiple functions'''

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

corpus, Filenames = loadData(
    DATA_PATH, isNotStopWord, lowerize, lemmatize)


# 2 - Build inverted index

def build_inverted_index(collection):
    '''builds the inverted index, and terms_per_document object'''
    print("Parsing files and building inverted index")
    filecount = len(collection)
    current = 0
    inverted_index = {}
    terms_per_document = len(list(collection.keys()))*[0]
    for document in collection:
        # Print progress
        current += 1
        if current % 100 == 0:
            stdout.write(f"Processing: {current} / {filecount}\r")
            stdout.flush()

        for term in collection[document]:
            terms_per_document[document] += 1
            if term in inverted_index:
                if document in inverted_index[term]:
                    inverted_index[term][document] += 1
                else:
                    inverted_index[term][document] = 1
            else:
                inverted_index[term] = {document: 1}

    stdout.write(f"Processing: {filecount} / {filecount}\r\n")
    stdout.flush()
    return inverted_index, terms_per_document


inverted_index, terms_per_document = build_inverted_index(corpus)

cache_info = lemmatize.cache_info()
cache_hit_ratio = 100 * cache_info.hits / (cache_info.hits + cache_info.misses)
print(
    f"lemmatize cache stats: {cache_info} (cache hit ratio: {cache_hit_ratio:.1f}%)")


# 3 - Save inverted index, posting data and file names

def save_inverted_index_pickle(inverted_index, filename):
    print("Saving inverted index to disk...")
    with open(filename, "wb") as f:
        pickle.dump(inverted_index, f)
        f.close()


save_inverted_index_pickle(inverted_index, "inverted_index")

# Exporting file names for post-processing
dump = json.dumps(Filenames)
f = open("Filenames.json", "w")
f.write(dump)
f.close()

# Exporting terms per document for vectorial models
dump = json.dumps(terms_per_document)
f = open("terms_per_document.json", "w")
f.write(dump)
f.close()
