import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from collections import OrderedDict

import pickle

# Fonction de chargement de la collection

def loadData(filename):
    with open(filename, 'r') as f:
        corpus = {}
        n = 0 # compteur pour donner un identifiant unique aux documents
        article = ""
        line = f.readline()
        while not re.match(r"\*STOP",line):
            if (re.match(r"^\*TEXT",line)): # C'est un début d'article
                doc_id = line.split()[1]
                line = f.readline()
                while not (re.match(r"\*TEXT",line) or re.match(r"\*STOP",line)):
                    article = article + " "+line.rstrip()
                    line = f.readline()
                n = n +1
                corpus[doc_id] = article
                article = ""
        print("{} articles ont été parsés".format(n))
        return(corpus)



# Segmentation d'un document avec split de python

def article_tokenize_simple(text):
    if type(text)!= str:
        raise Exception("The function takes a string as input data")
    else:
        return text.split() 




# Segmentation d'un document avec word_tokenize de nltk

def article_word_tokenize(article):
    if type(text)!= str:
        raise Exception("The function takes a string as input data")
    else:
        tokens=word_tokenize(text)
        return tokens


# Segmentation d'un document avec le RegexpTokenizer de nltk



def article_tokenize_other(text):
    if type(text)!= str:
        raise Exception("The function takes a string as input data")
    else:
        # On extrait les abbreviations
        tokenizer = RegexpTokenizer('[a-zA-Z]\.[a-zA-Z]')
        tokens = tokenizer.tokenize(text)
        # On extrait les mots et les nombres
        tokenizer = RegexpTokenizer('[a-zA-Z]{2,}|\d+\S?\d*')
        wordtokens = tokenizer.tokenize(text)
        return tokens + wordtokens


# Fonctions pour segmenter l'ensemble du corpus

def tokenize_Regexp_corpus(corpus):
    tokenized_corpus = {}
    for i in corpus:
        tokenized_corpus[i] = article_tokenize_other(corpus[i])
    return tokenized_corpus


def tokenize_WT_corpus(corpus):
    tokenized_corpus = {}
    for i in corpus:
        tokenized_corpus[i] = article_word_tokenize(corpus[i])
    return tokenized_corpus


def tokenize_simple_corpus(corpus):
    tokenized_corpus = {}
    for i in corpus:
        tokenized_corpus[i] = article_tokenize_simple(corpus[i])
    return tokenized_corpus


# Filtrage

# Fonction permettant de charger le vocabulaire de mots vides

def load_stop_word(filename):
    with open(filename, 'r') as f:
        stop_words = []
        line = f.readline()
        while line !='Z\n':
            if line !='\n':
                stop_words.append(line.rstrip())
            line = f.readline()
        stop_words.append(line.rstrip())
    return stop_words


# Fonction permettant de filtrer la collection des mots vides
def remove_stop_words(collection ,stop_word_file):
    collection_filtered={}
    for i in collection:
        collection_filtered[i]=[]
        for j in collection[i]:
            if j not in stop_word_file:
                collection_filtered[i].append(j)
    return collection_filtered



# Fonctions de normalisation



# Racinisation
def collection_stemming(segmented_collection):
    stemmed_collection={}
    stemmer = PorterStemmer () # initialisation d'un stemmer
    for i in segmented_collection:
        stemmed_collection[i]=[]
        for j in segmented_collection[i]:
            stemmed_collection[i].append( stemmer.stem(j))
    return stemmed_collection


# Lemmatisation



def collection_lemmatize(segmented_collection):
    lemmatized_collection={}
    stemmer = WordNetLemmatizer() # initialisation d'un lemmatiseur
    for i in segmented_collection:
        lemmatized_collection[i]=[]
        for j in segmented_collection[i]:
            lemmatized_collection[i].append( stemmer.lemmatize(j))
    return lemmatized_collection





# Fonctions pour l'index inversé


# Construction de l'index inversé



def build_inverted_index(collection,type_index):
    # On considère ici que la collection est pré-traitée
    inverted_index=OrderedDict()
    if type_index == 1:
        for document in collection:
            for term in collection[document]:
                if term in inverted_index.keys():
                    if document not in inverted_index[term]:
                        inverted_index[term].append(document)
                else:
                    inverted_index[term]=[document]
    elif type_index ==2:
        for document in collection:
            for term in collection[document]:
                if term in inverted_index.keys():
                    if document in inverted_index[term].keys():
                        inverted_index[term][document] = inverted_index[term][document] + 1
                    else:
                        inverted_index[term][document]= 1
                else:
                    inverted_index[term]=OrderedDict()
                    inverted_index[term][document]=1
    elif type_index==3:
        for document in collection:
            n=0
            for term in collection[document]:
                n = n+1
                if term in inverted_index.keys():
                    if document in inverted_index[term].keys():
                        inverted_index[term][document][0] = inverted_index[term][document][0] + 1
                        inverted_index[term][document][1].append(n)
                    else:
                        inverted_index[term][document]= [1,[n]]
                else:
                    inverted_index[term]=OrderedDict()
                    inverted_index[term][document]=[1,[n]]
                    
    return inverted_index


# Sauvergarde de l'index inversé sous la forme d'un fichier .txt

def save_inverted_index(inverted_index, filename, type_index):
    with open(filename, 'w') as f:
        for term in inverted_index:
            if type_index==1:
                f.write(term + "," + str(len(inverted_index[term])))
                for doc in inverted_index[term]:
                    f.write("\t" + doc)
                f.write("\n")
            if type_index==2:
                f.write(term + "," + str(len(inverted_index[term])))
                for doc in inverted_index[term]:
                    f.write("\t" + doc + "," + str(inverted_index[term][doc]))
                f.write("\n")
            if type_index==3:
                f.write(term + "," + str(len(inverted_index[term])))
                for doc in inverted_index[term]:
                    f.write("\t" + doc + "," + str(inverted_index[term][doc][0]) + ";")
                    for pos in inverted_index[term][doc][1]:
                        f.write(str(pos) + ",")
                f.write("\n")
        f.close()


# Chargement de l'index

def convert_list_to_int(liste):
    new_list = []
    for i in liste:
         new_list.append (int(i))
    return new_list
        



def load_inverted_index(filename,type_index):
    with open(filename, 'r') as f:
        inverted_index = OrderedDict()
        line = f.readline()
        while line!="":
            if type_index==1:
                line = line.rstrip()
                content = line.split("\t")
                term = content[0].split(",")[0]
                postings = content[1:]
                inverted_index[term] = postings
                line = f.readline()
            if type_index==2:
                line = line.rstrip()
                content = line.split("\t")
                term = content[0].split(",")[0]
                postings = content[1:]
                postings_with_tf = OrderedDict()
                for occurence in postings:
                    content = occurence.split(",")
                    postings_with_tf[content[0]] = int(content[1])
                inverted_index[term] = postings_with_tf
                line = f.readline()
            if type_index==3:
                line = line.rstrip()
                content = line.split("\t")
                term = content[0].split(",")[0]
                postings = content[1:]
                postings_with_tf_and_pos= OrderedDict()
                for occurence in postings:
                    content = occurence.split(";")
                    positions = content[1].rstrip(",").split(",")
                    positions=convert_list_to_int(positions)
                    document = content[0].split(",")
                    postings_with_tf_and_pos[document[0]] = [int(document[1]), positions]
                inverted_index[term] = postings_with_tf_and_pos
                line = f.readline()        
        f.close()         
        return inverted_index

# Avec pickle


# Ecriture sur disque
def save_inverted_index_pickle(inverted_index, filename):
     with open(filename, "wb") as f:
            pickle.dump(inverted_index,f)
            f.close()
    

# Chargement

def load_inverted_index_pickle(filename):
    with open(filename, 'rb') as fb:
        index = pickle.load(fb)
        return index


