import pickle


def save(obj, fpath):
    pickle.dump(obj, open(fpath, "wb"))


def load(fpath):
    return pickle.load(open(fpath, "rb"))
