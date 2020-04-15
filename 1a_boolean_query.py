import traceback
from tt import BooleanExpression
from utils import load_inverted_index_pickle, loadFilenames
from query import process, loadQueries, loadExpectedOutputs, DisplayMetrics

# Config

# The following parameter decides which boolean to use when combining the
# different words of the query
AND_vs_OR = "AND"

# 1 - Loading the inverted Index

inverted_index = load_inverted_index_pickle("inverted_index")

# 2 - Parsing and formatting queries
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
                                                 transformation_lem_query_to_boolean(query, AND_vs_OR))
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


loadedFiles = loadFilenames()

# Load expected output

ExpectedOutputs = loadExpectedOutputs(loadedFiles)

n = len(loadedFiles)

if (len(ExpectedOutputs) != len(Outputs) or len(Outputs) != len(Queries)):
    print("Output sizes not matching.")
else:
    for idx, query in enumerate(Queries):
        if len(query) > 0:
            print("Query", idx, ":", query, "found",
                  len(Outputs[idx]), "documents.")
            DisplayMetrics(ExpectedOutputs[idx], Outputs[idx], n)
