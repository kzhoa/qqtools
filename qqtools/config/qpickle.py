import pickle


def save_pickle(obj, fpath):
    pickle.dump(obj, open(fpath, "wb"))


def load_pickle(fpath):
    return pickle.load(open(fpath, "rb"))
