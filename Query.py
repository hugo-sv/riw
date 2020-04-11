import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from tt import BooleanExpression
import traceback


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
                    query.append(stemmer.lemmatize(word.upper()))
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


def transformation_query_to_postfixe(query):
    b = BooleanExpression(' '.join(query))
    return b.postfix_tokens
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

for query in Queries:
    try:
        q = transformation_query_to_postfixe(
            transformation_lem_query_to_boolean(query))
        out = processing_boolean_query_with_inverted_index(
            ['AND', 'OR', 'NOT'], q, inverted_index)
        print(f"### Query {query}: OK ###")
        print(out)
        print()
    except Exception:
        print(f"### Query {query}: failed ###")
        print(traceback.format_exc())

# 6 - Afficher Résultats

# Cf exemples dans Queries/ -> Liste ordonnée de n (?) documents

# 7 - Evaluer résultats

# Cf Queries/dev_output
