import pickle
import json


def load_inverted_index_pickle(filename):
    with open(filename, 'rb') as fb:
        index = pickle.load(fb)
        return index


def loadTermsPerDocument():
    terms_per_document = []
    with open("terms_per_document.json", 'r') as f:
        terms_per_document = json.load(f)
    return terms_per_document


def loadFilenames():
    loadedFiles = []
    with open("Filenames.json", 'r') as f:
        loadedFiles = json.load(f)
    return loadedFiles
