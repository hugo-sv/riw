import pickle
import sys
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import traceback
import json
import math
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# 1 - Loading the inverted Index and terms_per_document data


def load_inverted_index_pickle(filename):
    with open(filename, 'rb') as fb:
        index = pickle.load(fb)
        return index


def loadTermsPerDocument():
    terms_per_document = []
    with open("terms_per_document.json", 'r') as f:
        terms_per_document = json.load(f)
    return terms_per_document


inverted_index = load_inverted_index_pickle("inverted_index")
terms_per_document = loadTermsPerDocument()

# 2 - Parsing and formatting queries


stemmer = WordNetLemmatizer()
stopWords = set(stopwords.words('english'))
def process(query):
    res = []
    tokenized = word_tokenize(query)
    for word in tokenized:
        if word not in stopWords:
            res.append(stemmer.lemmatize(word.lower()))
    return res


def loadQueries():
    Queries = []
    default_queries_path = "Queries/dev_queries"
    shouldScore = True

    args = sys.argv[1:]
    if len(args) > 0:
        print("query detected in CLI argument -- skipping default queries")
        # disable output scoring:
        # we have nothing to compare the results of these queries against
        shouldScore = False
        for query in args:
            Queries.append(process(query))
    else:
        print(f"no query passed in CLI argument -- loading from {default_queries_path}")
        i = 0
        for i in range(1, 9):
            with open(f"{default_queries_path}/query.{i}", 'r') as f:
                query = process(f.read())
                Queries.append(query)

    return Queries, shouldScore


Queries, shouldScore = loadQueries()


# 3 - Executing queries
def tf(term, posting, inverted_index, terms_per_document):
    if inverted_index.get(term) is None:
        return 0
    a = inverted_index[term][posting]
    return a/terms_per_document[posting]


def df(term, inverted_index):
    if inverted_index.get(term) is None:
        return 0
    return len(inverted_index[term])


def idf(term, inverted_index, terms_per_document):
    N = len(terms_per_document)
    # Adding +1 to avoid division by 0 when looking for unknown words
    return math.log(N/(df(term, inverted_index)+1))


def tf_idf(term, posting, inverted_index, terms_per_document):
    return tf(term, posting, inverted_index, terms_per_document)*idf(term, inverted_index, terms_per_document)


def GetPostingsScore(query, terms_per_document, inverted_index):
    '''terms_per_document is a dictionary giving the number of terms per documents'''
    N = len(terms_per_document)
    query_norm = 0
    Scores = {}
    Document_norm = {}
    for j in range(N):
        Scores[j] = 0
        Document_norm[j] = 0
    for term in query:
        # Using tf-idf to compute term weight in the query.
        term_query_weight = (1/len(query)) * idf(term, inverted_index,
                                                 terms_per_document)
        query_norm += term_query_weight**2
        L = list(inverted_index[term.lower()].keys())
        for posting in L:
            # Using tf-idf to compute term weight in the document
            term_document_weight = tf_idf(
                term, posting, inverted_index, terms_per_document)
            Document_norm[posting] += term_document_weight**2
            # Updating score
            Scores[posting] += term_query_weight*term_document_weight
    # Normalizing each scores
    for j in range(N):
        if Scores[j] != 0:
            Scores[j] /= math.sqrt(Document_norm[j])*math.sqrt(query_norm)
    return Scores


def GetSortedBestDocuments(Scores, threshold):
    # Return document ids having a score > threshold, sorted by score
    FilteredScores = dict(
        filter(lambda elem: elem[1] > threshold, Scores.items()))
    return [k for k, v in sorted(FilteredScores.items(), key=lambda item: item[1])]


Outputs = []
Threshold = 0.5
for query in Queries:
    if len(query) > 0:
        try:
            Scores = GetPostingsScore(
                query, terms_per_document, inverted_index)
            out = GetSortedBestDocuments(Scores, Threshold)
            Outputs.append(out)
        except Exception:
            print(f"### Query {query} : failed ###")
            print(traceback.format_exc())
            Outputs.append([])
    else:
        Outputs.append([])

# 4 - Evaluate results

if not shouldScore:
    for i, query in enumerate(Queries):
        print(f"\n### Query {query} ###")
        print(Outputs[i])
    exit()

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


def DisplayMetrics(expected, actual, n):
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
    print("Output sizes not matching : ", len(
        ExpectedOutputs), len(Outputs), len(Queries))
else:
    for idx, query in enumerate(Queries):
        if len(query) > 0:
            print("Query", idx, ":", query, "found",
                  len(Outputs[idx]), "documents.")
            DisplayMetrics(
                ExpectedOutputs[idx], Outputs[idx], n)


# Plotting ROC curves

def PlotRoc2(Queries, ExpectedOutputs, terms_per_document, inverted_index):
    plt.figure()
    lw = 2

    for idx, query in enumerate(Queries):
        if len(query) > 0:
            Scores = GetPostingsScore(
                query, terms_per_document, inverted_index)
            y_score = [Scores[i] for i in range(len(terms_per_document))]
            y_expected = [0] * len(terms_per_document)
            for document in ExpectedOutputs[idx]:
                y_expected[document] = 1
            fpr, tpr, _ = roc_curve(y_expected, y_score)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr,
                     lw=lw, label=str(query)+'(area = %0.2f)' % roc_auc)

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC CURVE')
    plt.legend(loc="lower right")
    plt.show()


PlotRoc2(Queries, ExpectedOutputs, terms_per_document, inverted_index)
