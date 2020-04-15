import json
import pickle
import sys
import traceback
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from tt import BooleanExpression

# 1 - Loading the inverted Index


def load_inverted_index_pickle(filename):
    with open(filename, 'rb') as fb:
        index = pickle.load(fb)
        return index


inverted_index = load_inverted_index_pickle("inverted_index")

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
        print(f"no query passed in CLI argument -- loading from {default_queries_path}")
        i = 0
        for i in range(1, 9):
            with open(f"{default_queries_path}/query.{i}", 'r') as f:
                query = process(f.read())
                Queries.append(query)
                fullQuery += query
        # Adding a query concatenating all previous queries
        Queries.append(list(set(fullQuery)))

    return Queries, shouldScore


Queries, shouldScore = loadQueries()


# 3 - Executing queries

# Boolean Mode

def transformation_lem_query_to_boolean(query, operator='AND'):
    boolean_query = []
    for token in query:
        boolean_query.append(token)
        boolean_query.append(operator)
    boolean_query.pop()
    return boolean_query


def transformation_query_to_postfixe(booleanOperators, query):
    # query can contain keywords that do not pass ttable's is_valid_identifier function:
    # https://github.com/welchbj/tt/blob/43beecac6fae9ac7e2035b66b6e790cfc25167ba/tt/definitions/operands.py#L33-L90
    # we still want to be able to handle queries containing forbidden keywords like "class"
    # so we make the identifiers abstract
    abstract_query = []
    matcher = {}
    for token in query:
        if token in booleanOperators:
            matcher[token] = token
            abstract_query.append(token)
        else:
            abstract_token = f"var{len(abstract_query)}"
            abstract_query.append(abstract_token)
            matcher[abstract_token] = token
    postfix_abstract = BooleanExpression(
        ' '.join(abstract_query)).postfix_tokens
    return [matcher[key] for key in postfix_abstract]

# AND operator


def merge_and_postings_list(posting_term1, posting_term2):
    result = []
    n = len(posting_term1)
    m = len(posting_term2)
    i = 0
    j = 0
    while i < n and j < m:
        if posting_term1[i] == posting_term2[j]:
            result.append(posting_term1[i])
            i = i+1
            j = j+1
        else:
            if posting_term1[i] < posting_term2[j]:
                i = i+1
            else:
                j = j+1
    return result

# OR operator


def merge_or_postings_list(posting_term1, posting_term2):
    result = []
    n = len(posting_term1)
    m = len(posting_term2)
    i = 0
    j = 0
    while i < n and j < m:
        if posting_term1[i] == posting_term2[j]:
            result.append(posting_term1[i])
            i = i+1
            j = j+1
        else:
            if posting_term1[i] < posting_term2[j]:
                result.append(posting_term1[i])
                i = i+1
            else:
                result.append(posting_term2[j])
                j = j+1
    return result


# AND NOT operator

def merge_and_not_postings_list(posting_term1, posting_term2):
    result = []
    n = len(posting_term1)
    m = len(posting_term2)
    i = 0
    j = 0
    while i < n and j < m:
        if posting_term1[i] == posting_term2[j]:
            i = i+1
            j = j+1
        else:
            if posting_term1[i] < posting_term2[j]:
                result.append(posting_term1[i])
                i = i+1
            else:
                j = j+1
    return result


def boolean_operator_processing_with_inverted_index(BoolOperator, posting_term1, posting_term2):
    result = []
    if BoolOperator == "AND":
        result.append(merge_and_postings_list(posting_term1, posting_term2))
    elif BoolOperator == "OR":
        result.append(merge_or_postings_list(posting_term1, posting_term2))
    elif BoolOperator == "NOT":
        result.append(merge_and_not_postings_list(
            posting_term1, posting_term2))
    return result

# Custom exception to handle queries querying unknown words


class MissingTerm(Exception):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message


def processing_boolean_query_with_inverted_index(booleanOperators, query, inverted_index):
    evaluation_stack = []
    for term in query:
        if term.upper() not in booleanOperators:
            if inverted_index.get(term.lower()) is not None:
                evaluation_stack.append(
                    list(inverted_index[term.lower()].keys()))
            else:
                raise MissingTerm(
                    "A term is missing form the inverted_index", term.lower())
        else:
            if term.upper() == "NOT":
                operande = evaluation_stack.pop()
                eval_prop = boolean_operator_processing_with_inverted_index(
                    term.upper(), evaluation_stack.pop(), operande)
                evaluation_stack.append(eval_prop[0])
                evaluation_stack.append(eval_prop[0])
            else:
                operator = term.upper()
                eval_prop = boolean_operator_processing_with_inverted_index(
                    operator, evaluation_stack.pop(), evaluation_stack.pop())
                evaluation_stack.append(eval_prop[0])
    return evaluation_stack.pop()


booleanOperators = ['AND', 'OR', 'NOT']
Outputs = []
for query in Queries:
    try:
        q, out = [], []
        if len(query) == 0:
            q = query
        else:
            q = transformation_query_to_postfixe(booleanOperators,
                                                 transformation_lem_query_to_boolean(query, 'AND'))
            out = processing_boolean_query_with_inverted_index(
                booleanOperators, q, inverted_index)
        Outputs.append(out)
    except MissingTerm as err:
        print(
            f"### Query {query}: failed : Missing term ( {err.message} ) ###")
        Outputs.append([])
    except Exception:
        print(f"### Query {query}: failed ###")
        print(traceback.format_exc())
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


ExpectedOutputs = loadExpectedOutputs(loadedFiles)


def compareOutputsBoolean(expected, actual, n):
    # With boolean queries, there are no notions of order
    TruePositives = 0
    for document in expected:
        if document in actual:
            TruePositives += 1

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
    print("Output sizes not matching.")
else:
    for idx, query in enumerate(Queries):
        if len(query) > 0:
            print("Query", idx, ":", query, "found",
                  len(Outputs[idx]), "documents.")
            compareOutputsBoolean(
                ExpectedOutputs[idx], Outputs[idx], n)
