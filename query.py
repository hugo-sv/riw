from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import sys

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
    fullQuery = []
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
        print(
            f"no query passed in CLI argument -- loading from {default_queries_path}")
        i = 0
        for i in range(1, 9):
            with open(f"{default_queries_path}/query.{i}", 'r') as f:
                query = process(f.read())
                Queries.append(query)
                fullQuery += query
        # Adding a query concatenating all previous queries
        Queries.append(list(set(fullQuery)))

    return Queries, shouldScore


def loadExpectedOutputs(loadedFiles):
    fileIdx = {}
    for idx, filename in enumerate(loadedFiles):
        fileIdx[filename] = idx
    MissingFiles = 0
    Outputs = []
    fullOutput = []
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
        fullOutput += current_output
    # We consider the full query expected output is a concatenation of all outputs
    Outputs.append(list(set(fullOutput)))
    print("Missing files :", MissingFiles)
    return Outputs


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


def DisplayResults(Queries, ExpectedOutputs, Outputs, loadedFiles):
    n = len(loadedFiles)

    if (len(ExpectedOutputs) != len(Outputs) or len(Outputs) != len(Queries)):
        print("Output sizes not matching : ", len(
            ExpectedOutputs), len(Outputs), len(Queries))
    else:
        for idx, query in enumerate(Queries):
            if len(query) > 0:
                print("Query", idx, ":", query, "found",
                      len(Outputs[idx]), "documents.")
                DisplayMetrics(ExpectedOutputs[idx], Outputs[idx], n)
