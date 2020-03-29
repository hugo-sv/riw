import pickle

def save_inverted_index_pickle(inverted_index, filename):
     with open(filename, "wb") as f:
            pickle.dump(inverted_index,f)
            f.close()


def load_inverted_index_pickle(filename):
    with open(filename, 'rb') as fb:
        index = pickle.load(fb)
        return index
