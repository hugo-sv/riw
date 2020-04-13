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

# 1 - Import de la collection


def loadData(dataPath):
    '''loadData Loads the dataset at the given path.
    It must be in the following format : '''
    print("Loading the dataset")
    Filenames = []
    corpus = {}
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
                    dirPath = join(dir, filename)
                    # Appending corpus, using Filenames' index
                    # Tokenizing with nltk here to save computation time
                    corpus[i] = word_tokenize(f.read())
                    Filenames.append(dirPath)
                    i += 1
                    if i >= 1000:
                        return corpus, Filenames
    return corpus, Filenames


dataPath = "Data/pa1-data/"
corpus, Filenames = loadData(dataPath)

json = json.dumps(Filenames)
f = open("Filenames.json", "w")
f.write(json)
f.close()

# 2 - Processing de la collection


def remove_stop_words(collection):
    '''remove_stop_words Remove stop words (from Lab1.py)'''
    print("Removing stop words")
    stopWords = set(stopwords.words('english'))
    collection_filtered = {}
    for i in collection:
        collection_filtered[i] = []
        for j in collection[i]:
            if j not in stopWords:
                collection_filtered[i].append(j)
    return collection_filtered


corpus = remove_stop_words(corpus)


def collection_stemming(segmented_collection):
    print("Stemming the collection")
    stemmed_collection = {}
    stemmer = PorterStemmer()  # initialisation d'un stemmer
    for i in segmented_collection:
        stemmed_collection[i] = []
        for j in segmented_collection[i]:
            stemmed_collection[i].append(stemmer.stem(j.lower()))
    return stemmed_collection


def collection_lemmatize(segmented_collection):
    print("Lemmatization of the collection")
    lemmatized_collection = {}
    stemmer = WordNetLemmatizer()  # initialisation d'un lemmatiseur
    for i in segmented_collection:
        lemmatized_collection[i] = []
        for j in segmented_collection[i]:
            lemmatized_collection[i].append(
                stemmer.lemmatize(j.lower()))
    return lemmatized_collection


# Apply lemmatization only, as it provides better results
corpus = collection_lemmatize(corpus)

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
