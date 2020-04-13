import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from tt import BooleanExpression
import traceback
import json


def load_inverted_index_pickle(filename):
    with open(filename, 'rb') as fb:
        index = pickle.load(fb)
        return index


inverted_index = load_inverted_index_pickle("inverted_index")

# print(inverted_index['science'])

# 4 - Parser les requetes


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

# print(Queries)

# 5 - Executer des requetes

# Mode Booléen


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

# Operateur AND sur posting listes


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

# Operateur OR sur posting listes


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


# Operateur AND NOT sur posting listes

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


def processing_boolean_query_with_inverted_index(booleanOperators, query, inverted_index):
    evaluation_stack = []
    for term in query:
        if term.upper() not in booleanOperators:
            evaluation_stack.append(inverted_index[term.lower()])
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
                                                 transformation_lem_query_to_boolean(query))
            out = processing_boolean_query_with_inverted_index(
                booleanOperators, q, inverted_index)
        print(f"### Query {query} -> {q}: OK ###")
        print(out)
        print()
        Outputs.append(out)
    except Exception:
        print(f"### Query {query}: failed ###")
        print(traceback.format_exc())
        Outputs.append([])

# 6 - Afficher Résultats

# Cf exemples dans Queries/ -> Liste ordonnée de n (?) documents

# 7 - Evaluer résultats

# Load expected output


def loadExpectedOutputs():
    loadedFiles = []
    with open("Filenames.json", 'r') as f:
        loadedFiles = json.load(f)

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


ExpectedOutputs = loadExpectedOutputs()
# print(ExpectedOutputs)


def compareOutputsBoolean(expected, actual):
    # With boolean queries, there are no notions of order
    if len(expected) > 0 and len(actual) > 0:
        intersected = len(set(expected).intersection(set(actual)))
        return intersected/len(expected)
    return 0


if (len(ExpectedOutputs) != len(Outputs) or len(Outputs) != len(Queries)):
    print("Issue in Ouputs size.")
else:
    for idx, query in enumerate(Queries):
        print("Query :", query)
        print("Score :", compareOutputsBoolean(
            ExpectedOutputs[idx], Outputs[idx]))
