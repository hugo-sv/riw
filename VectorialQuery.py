import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import traceback
import json
import math
import matplotlib.pyplot as plt

# 1 - Loading the inverted Index and postings data


def load_inverted_index_pickle(filename):
    with open(filename, 'rb') as fb:
        index = pickle.load(fb)
        return index


def loadPostings():
    postings = []
    with open("Postings.json", 'r') as f:
        postings = json.load(f)
    return postings


inverted_index = load_inverted_index_pickle("inverted_index")
postings = loadPostings()

# 2 - Parsing and formating queries


def loadQueries():
    Queries = []
    stemmer = WordNetLemmatizer()
    stopWords = set(stopwords.words('english'))
    i = 0
    for i in range(1, 9):
        with open("Queries/dev_queries/query."+str(i), 'r') as f:
            raw_query = f.read()
            # Tokenize
            tokenized_query = word_tokenize(raw_query)
            query = []
            for word in tokenized_query:
                # Remove Stop words
                if word not in stopWords:
                    # Lemmatization
                    query.append(stemmer.lemmatize(word.lower()))
            Queries.append(query)
    return Queries


Queries = loadQueries()


# 3 - Executing queries
def tf(term, posting, inverted_index, postings):
    a = inverted_index[term][posting]
    return a/postings[posting]


def df(term, inverted_index):
    return len(inverted_index[term])


def idf(term, inverted_index, postings):
    N = len(postings)
    # We could use math.log(N/(df(term, inverted_index)+1)) to avoid division by 0
    return math.log(N/df(term, inverted_index))


def tf_idf(term, posting, inverted_index, postings):
    return tf(term, posting, inverted_index, postings)*idf(term, inverted_index, postings)


def GetPostingsScore(query, postings, inverted_index):
    '''postings is a dictionary giving the number of terms per documents'''
    N = len(postings)
    n_q = 0
    Scores = {}
    Norm = {}
    for j in range(N):
        Scores[j] = 0
        Norm[j] = 0
    for term in query:
        term_query_weight = idf(term, inverted_index, postings) * 1/len(query)
        n_q += term_query_weight*term_query_weight
        L = list(inverted_index[term.lower()].keys())
        for posting in L:
            # Using tf-idf to compute term weight in the document
            term_document_weight = tf_idf(
                term, posting, inverted_index, postings)
            Scores[posting] += term_query_weight*term_document_weight
            Norm[posting] += term_document_weight*term_document_weight
    # Normalizing each scores
    for j in range(N):
        if Scores[j] != 0:
            # Normalize
            Scores[j] *= 1/(math.sqrt(Norm[j])*math.sqrt(n_q))
    return Scores


def GetSortedBestDocuments(Scores, threshold):
    # Return document ids having a score > threshold, sorted by score
    FilteredScores = dict(
        filter(lambda elem: elem[1] > threshold, Scores.items()))
    return [k for k, v in sorted(FilteredScores.items(), key=lambda item: item[1])]


def GetFilteredBestDocuments(Scores, threshold):
    # Return document ids having a score > threshold
    return list(dict(
        filter(lambda elem: elem[1] > threshold, Scores.items())).keys())


Outputs = []
bestScore = 0
for query in Queries:
    if len(query) > 0:
        try:
            Scores = GetPostingsScore(query, postings, inverted_index)
            out = GetSortedBestDocuments(Scores, 0.5)
            bestScore = max(bestScore, max(Scores.values()))
            print(f"### Query {query} : OK with {len(out)} results ###")
            Outputs.append(out)
        except Exception:
            print(f"### Query {query} : failed ###")
            print(traceback.format_exc())
            Outputs.append([])
    else:
        Outputs.append([])
print("Best Score = ", bestScore)

# 4 - Evaluate results


def loadFilenames():
    loadedFiles = []
    with open("Filenames.json", 'r') as f:
        loadedFiles = json.load(f)
    return loadedFiles


loadedFiles = loadFilenames()

# Load expected output


def loadExpectedOutputs(loadedFiles):
    fileIdx = {}
    for idx, filename in enumerate(loadedFiles):
        fileIdx[filename] = idx
    MissingFiles = 0
    Outputs = []
    i = 0
    for i in range(1, 9):
        current_output = []
        with open(f"Queries/dev_output/{i}.out", 'r') as f:
            for line in f:
                parsed_line = line.rstrip('\n')
                if parsed_line in fileIdx:
                    current_output.append(fileIdx[parsed_line])
                else:
                    MissingFiles += 1
        Outputs.append(current_output)
    print("Missing files :", MissingFiles)
    return Outputs


ExpectedOutputs = loadExpectedOutputs(loadedFiles)


def compareOutputsBoolean(expected, actual, n):
    TruePositives = len(set(actual).intersection(set(expected)))

    precision, recall, f1 = 0, 0, 0
    if len(actual) > 0:
        precision = TruePositives/len(actual)

    if len(expected) > 0:
        recall = TruePositives/len(expected)

    if recall+precision > 0:
        f1 = 2*(precision*recall)/(precision+recall)

    accuracy = (TruePositives + (n - len(actual) -
                                 len(expected) + TruePositives))/n
    print('\t Precision = {:.2f}'.format(precision))
    print('\t Recall = {:.2f}'.format(recall))
    print('\t Accuracy = {:.2f}'.format(accuracy))
    print('\t F1 Score = {:.2f}'.format(f1))
    return


n = len(loadedFiles)

if (len(ExpectedOutputs) != len(Outputs) or len(Outputs) != len(Queries)):
    print("Ouput sizes not matching : ", len(
        ExpectedOutputs), len(Outputs), len(Queries))
else:
    for idx, query in enumerate(Queries):
        if len(query) > 0:
            print("Query", idx, ":", query, "found",
                  len(Outputs[idx]), "documents.")
            compareOutputsBoolean(
                ExpectedOutputs[idx], Outputs[idx], n)


# Plotting ROC curves

def getSpecificityRecall(expected, actual, n):
    TruePositives = len(set(actual).intersection(set(expected)))

    specificity, recall = 0, 0
    if n-len(expected) > 0:
        specificity = (len(actual)-TruePositives)/(n-len(expected))

    if len(expected) > 0:
        recall = TruePositives/len(expected)

    return specificity, recall


def PlotRoc(Queries, postings, inverted_index, bestScore):
    print("Computing and plotting ROC curves ...")
    Query_Scores = {}
    Thresholds = [-1]
    for idx, query in enumerate(Queries):
        if len(query) > 0:
            Scores = GetPostingsScore(query, postings, inverted_index)
            Query_Scores[idx] = Scores
            Thresholds = list(set(Thresholds + list(Scores.values())))
            Thresholds.sort()
    while len(Thresholds) % 10 != 0:
        # So that we can go 10 by 10
        Thresholds.append(2)
    print("Thresholds :", len(Thresholds))
    for idx, query in enumerate(Queries):
        print("Progress :", idx, "/", len(Queries))
        if len(query) > 0:
            FPR, TPR = [], []
            Scores = Query_Scores[idx]
            i = 0
            for threshold in Thresholds[::10]:
                if i % 300 == 1:
                    print(i)
                out = GetFilteredBestDocuments(Scores, threshold)
                expected = ExpectedOutputs[idx]
                specificity, recall = getSpecificityRecall(expected, out, n)
                FPR.append(1-specificity)
                TPR.append(recall)
                i += 1
            plt.plot(FPR, TPR, label=str(idx))
    print("Plot finished")
    plt.legend()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.show()


PlotRoc(Queries, postings, inverted_index, bestScore)
