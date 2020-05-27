import os
import numpy as np
import pandas as pd
import io
import pickle

def read_embeddings(emb_path):
    if emb_path[-3:] == "csv":
        return read_csv_embeddings(emb_path)
    else:
        return read_txt_embeddings(emb_path)

# reads space separated .csv embeddings file, without first metadata row
# outputs embeddings as a numpy array, and two word2id and id2word maps
def read_csv_embeddings(emb_path):
    words2ids = dict()
    if os.path.isfile(emb_path):  # if we have saved embeddings file
        df = pd.read_csv(emb_path, sep=' ', header=None, converters={0 : str}) # because 'null'/'nan' can be among words
        vectors = df.values[:, 1:].astype(np.float64)
        for index, row in df.iterrows():
            words2ids[row[0]] = index
        ids2words = dict(zip(words2ids.values(), words2ids.keys()))
        return vectors, words2ids, ids2words


# reads space separated .txt embeddings file, WITH first metadata row 'vocab_size dimensionality' e.g. 1000 300
# outputs embeddings as a numpy array, and two word2id and id2word maps
def read_txt_embeddings(emb_path):
    """
    Reload pretrained embeddings from a text file.
    """
    words2ids = {}
    vectors = None

    # load pretrained embeddings
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        for i, line in enumerate(f):
            if i == 0:
                split = line.split()
                vectors = np.empty([int(split[0]), int(split[1])])
            else:
                word, vect = line.rstrip().split(' ', 1)
                vect = np.fromstring(vect, sep=' ')
                vectors[len(words2ids)] = vect
                words2ids[word] = len(words2ids)

    ids2words = dict(zip(words2ids.values(), words2ids.keys()))
    return vectors, words2ids, ids2words


def read_txt_embeddings_test(emb_path):
    """
    Reload pretrained embeddings from a text file.
    """
    words2ids = {}
    vectors = None

    # load pretrained embeddings
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        for i, line in enumerate(f):
            if i > 10:
                break
            if i == 0:
                split = line.split()
                vectors = np.empty([int(split[0]), int(split[1])])
            else:
                word, vect = line.rstrip().split(' ', 1)
                vect = np.fromstring(vect, sep=' ')
                vectors[len(words2ids)] = vect
                words2ids[word] = len(words2ids)

    ids2words = dict(zip(words2ids.values(), words2ids.keys()))
    return vectors, words2ids, ids2words



# reads sentences from .txt file into a generator containing lists of words for each sentence
# use as is when you need to iterate over it just once
# use read_sentences_list when you need to iterate more then once
def read_sentences(filepath):
    for line in open(filepath, 'r', encoding="utf8"):
        yield line.split()


# reads sentences from .txt file into a list of lists of words
def read_sentences_list(filepath):
    return list(read_sentences(filepath))


# load dictionary from file as a dict object (multiple translations are not supported)
def load_dictionary(filepath):
    dictionary = {}
    with io.open(filepath, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        for i, line in enumerate(f):
            word, translation = line.rstrip().split(' ', 1)
            dictionary[word] = translation

    reverse_dict = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dict


# load dictionary from file as list of word-translation pairs (arrays)
def load_dictionary_list(filepath):
    dictionary = []
    with io.open(filepath, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        for i, line in enumerate(f):
            word, translation = line.rstrip().split(' ', 1)
            dictionary.append([word,translation])
    return dictionary


def normalize_embeddings(emb):
    norm = np.linalg.norm(emb, axis=1)
    nonzero = norm > 0
    emb[nonzero] /= norm[nonzero] [:, None]
    return emb


# read monolingual dict, of the form word-# of times it appeared in corpus and its position
def load_freq_dict(filepath):
    dictionary = {}
    with io.open(filepath, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        for i, line in enumerate(f):
            word, n_times, position = line.rstrip().split(' ', 2)
            dictionary[word] = [int(n_times), int(position)]
    return dictionary


# reads the list of words and breaks them into categories: common, rare, not found
def break_freq_bins(words, dict, limit):
    common_words = []
    rare_words = []
    not_found = []

    for word in words:
        freq = dict.get(word)
        if freq is None:
            not_found.append(word)
        elif freq[1] < limit:
            common_words.append([word, freq])
        else:
            rare_words.append([word, freq])

    common_words.sort(key=lambda x: x[1][1])
    rare_words.sort(key=lambda x: x[1][1])

    print("COMMON WORDS:")
    for word in common_words:
        print(word[0], word[1])

    print("RARE WORDS:")
    for word in rare_words:
        print(word[0], word[1])

    print("NOT FOUND:")
    for word in not_found:
        print(word)


# converts csv file to txt w2v standard format, with vocab count and dim on top
def change_emb_format(embs_file, results_file):
    if embs_file[-3:] == 'csv':
        with open(embs_file, 'r') as f:
            lines = f.readlines()
            n_vocab = len(lines)
            dim = len(lines[0].split(' ')) - 1
        with open(results_file, 'w') as output:
            output.write(str(n_vocab) + ' ' + str(dim) + '\n')
            with open(embs_file, 'r') as f:
                temp = f.read()
            output.write(temp)



def find_common():
    p = '/media/HDD/public/vitaly/segmented-embeddings/embeddings_full/txt_format/'
    p1 = p + "ru_skipgram_100000"
    p2 = p + "ru_fasttext_100000"
    p3 = p + "ru_morph_100000"

    e1, words2ids1, ids2words1 = read_embeddings(p1+ ".txt")
    e2, words2ids2, ids2words2 = read_embeddings(p2+ ".txt")
    e3, words2ids3, ids2words3 = read_embeddings(p3+ ".txt")
    common_words = set(words2ids1.keys()).intersection(set(words2ids2.keys()), set(words2ids3.keys()))

    cut1 = open(p1 + "_cut.txt", "w")
    cut2 = open(p2 + "_cut.txt", "w")
    cut3 = open(p3 + "_cut.txt", "w")

    cut1.write("%d %d\n" % (len(common_words), 300))
    cut2.write("%d %d\n" % (len(common_words), 300))
    cut3.write("%d %d\n" % (len(common_words), 300))

    for w in common_words:
        cut1.write("%s %s\n" % (w, " ".join(list(map(str, e1[words2ids1[w]])))))
        cut2.write("%s %s\n" % (w, " ".join(list(map(str, e2[words2ids2[w]])))))
        cut3.write("%s %s\n" % (w, " ".join(list(map(str, e3[words2ids3[w]])))))

#find_common()



# change_emb_format("../common_data/word2vec/rus_9M_merged_v600000_w5/emb.csv", "emb.txt")
