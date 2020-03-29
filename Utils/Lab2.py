from tt import BooleanExpression

from Utils.Lab1 import *
from collections import Counter
from collections import *

# Modèle booleen

# Transformation d'une requete en langage naturel sous sa forme logique

def transformation_query_to_boolean(query):
    boolean_query=[]
    for token in query.split():
        boolean_query.append(token)
        boolean_query.append('AND')
    boolean_query.pop()
    return boolean_query

def transformation_lem_query_to_boolean(query):
    boolean_query=[]
    for token in query:
        boolean_query.append(token)
        boolean_query.append('AND')
    boolean_query.pop()
    return boolean_query


# Transformation d'une requête en notation polonaise inversée


def transformation_query_to_postfixe(query):
    b = BooleanExpression(query)
    return b.postfix_tokens


# Operateur AND sur posting listes
def merge_and_postings_list(posting_term1,posting_term2):
    result=[]
    n = len(posting_term1)
    m = len(posting_term2)
    i = 0
    j = 0
    while i < n and j <m:
        if posting_term1[i] == posting_term2[j]:
            result.append(posting_term1[i])
            i = i+1
            j = j+1
        else:
            if posting_term1[i] < posting_term2[j]:
                i = i+1
            else:
                j=j+1
    return result

# Operateur OR sur posting listes

def merge_or_postings_list(posting_term1,posting_term2):
    result=[]
    n = len(posting_term1)
    m = len(posting_term2)
    i = 0
    j = 0
    while i < n and j <m:
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
                j=j+1
    return result


# Operateur AND NOT sur posting listes

def merge_and_not_postings_list(posting_term1,posting_term2):
    result=[]
    n = len(posting_term1)
    m = len(posting_term2)
    i = 0
    j = 0
    while i < n and j <m:
        if posting_term1[i] == posting_term2[j]:
            i = i+1
            j = j+1
        else:
            if posting_term1[i] < posting_term2[j]:
                result.append(posting_term1[i])
                i = i+1
            else:
                j=j+1
    return result

# Fonction generale

def boolean_operator_processing_with_inverted_index(BoolOperator,posting_term1,posting_term2):
    result=[]
    if BoolOperator == "AND":
        result.append(merge_and_postings_list(posting_term1,posting_term2))
    elif BoolOperator=="OR" :
        result.append(merge_or_postings_list(posting_term1,posting_term2))
    elif BoolOperator == "NOT":
        result.append(merge_and_not_postings_list(posting_term1,posting_term2))
    return result


# Traitement d'une requête booleenne

def processing_boolean_query_with_inverted_index(booleanOperator,query, inverted_index):
    relevant_docs = {}
    evaluation_stack = []
    for term in query:
        if term.upper() not in booleanOperator:
            evaluation_stack.append(inverted_index[term.upper()])
        else:
            if term.upper() == "NOT":
                operande= evaluation_stack.pop()
                eval_prop = boolean_operator_processing_with_inverted_index(term.upper(), evaluation_stack.pop(),operande)
                evaluation_stack.append(eval_prop[0])
                evaluation_stack.append(eval_prop[0])
            else:
                operator = term.upper()
                eval_prop =  boolean_operator_processing_with_inverted_index(operator, evaluation_stack.pop(),evaluation_stack.pop())
                evaluation_stack.append(eval_prop[0])
    return  evaluation_stack.pop()



#  MODELE VECTORIEL

# Pre-traitement requête


def remove_non_index_term(query,inverted_index):
    query_filt=[]
    for token in query:
        if token in inverted_index:
            query_filt.append(token)
    return query_filt



def pre_processed_query(query,inverted_index):
    tokenized_query = article_tokenize_other(query)
    filt_query = remove_non_index_term(tokenized_query,inverted_index)
    filtered_query = remove_stop_words({"query":filt_query},load_stop_word("./Data/Time/TIME.STP"))
    normalized_query = collection_lemmatize(filtered_query)
    return normalized_query["query"]







# Fonctions pour les schémas de ponderation


def get_tf(term,doc_ID,index_frequence):
    return index_frequence[term][doc_ID]



def get_tf_logarithmique (term,doc_ID, index_frequence):
    tf = get_tf(term,doc_ID, index_frequence)
    if tf > 0:
        return 1 +log(tf)
    else:
        return 0




def get_stats_document(document):
    counter= Counter()
    for term in document:
        counter.update([term])
    stats={}
    stats["freq_max"] = counter.most_common(1)[0][1]
    stats["unique_terms"] = len(counter.items())
    tf_moy = sum(counter.values())
    stats["freq_moy"] = tf_moy/len(counter.items())
    return stats



def get_stats_collection(collection):
    stats={}
    stats["nb_docs"]=len(collection.keys())
    for doc in collection:
        stats[doc] = get_stats_document(collection[doc])
    return stats


def get_tf_normalise(term, doc_ID, index_frequence, stats_collection):
    tf = get_tf(term, doc_ID, index_frequence)
    tf_normalise = 0.5 + 0.5 * (tf / stats_collection[doc_ID]["freq_max"])
    return tf_normalise


from math import *

def get_tf_logarithme_normalise(term,doc_ID, index_frequence,stats_collection):
        tf = get_tf(term,doc_ID, index_frequence)
        tf_logarithme_normalise = (1 +log(tf))/(1 + log(stats_collection[doc_ID]["freq_moy"]))
        return tf_logarithme_normalise


def get_idf(term,index_frequence,nb_doc):
    return log(nb_doc/len(index_frequence[term].keys()))






def processing_vectorial_query(query, inverted_index, stats_collection, weighting_scheme_document,weighting_scheme_query):
    relevant_docs = {}
    counter_query= Counter()
    query_pre_processed = pre_processed_query(query,inverted_index)
    print(query)
    nb_doc = stats_collection["nb_docs"]
    norm_query=0.
    for term in query_pre_processed:
        if term in inverted_index:
            w_term_query=0.
            counter_query.update([term])
            if weighting_scheme_query=="binary":
                w_term_query = 1
            if weighting_scheme_query=="frequency":
                w_term_query = counter_query[term]
            norm_query = norm_query + w_term_query*w_term_query
            for doc in inverted_index[term]:
                w_term_doc = 0.
                relevant_docs[doc]=0.
                if weighting_scheme_document=="binary":
                    w_term_doc=1
                if weighting_scheme_document=="frequency":
                    w_term_doc = get_tf(term,doc,inverted_index)
                if weighting_scheme_document=="tf_idf_normalize":
                    w_term_doc = get_tf_normalise(term,doc, inverted_index,stats_collection)*get_idf(term,inverted_index,nb_doc)
                if weighting_scheme_document=="tf_idf_logarithmic":
                    w_term_doc = get_tf_logarithmique (term,doc, inverted_index)*get_idf(term,inverted_index,nb_doc)
                if weighting_scheme_document=="tf_idf_logarithmic_normalize":
                    w_term_doc = get_tf_logarithme_normalise (term,doc, inverted_index,stats_collection)*get_idf(term,inverted_index,nb_doc)
                relevant_docs[doc] = relevant_docs[doc] + w_term_doc*w_term_query
    ordered_relevant_docs = OrderedDict(sorted(relevant_docs.items(), key=lambda t: t[1], reverse=True))
    return ordered_relevant_docs