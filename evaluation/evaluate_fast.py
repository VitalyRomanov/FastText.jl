import numpy as np
from utils import read_csv_embeddings, read_txt_embeddings, normalize_embeddings, read_sentences_list
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr, pearsonr
import sys

QS_PATH = './analogy_questions/'
QS_FILE_TT = QS_PATH + 'tat_analogies.txt'
QS_FILE_RU_SEM = QS_PATH + 'ru_semantic.txt'
QS_FILE_RU_SYN = QS_PATH + 'ru_synthetic.txt'

QS_FILE_EN = QS_PATH + 'en_analogies.txt'
QS_FILE_EN_BATS = QS_PATH + 'bats.txt'

SIM_PATH_TT = './similarity/similarity.csv'
REL_PATH_TT = './relatedness/relatedness.csv'

SIMLEX999_EN = './similarity/en_simlex999.csv'
SIMLEX999_RU = './similarity/ru_simlex999.csv'
SIMLEX965_RU = './similarity/ru_simlex965.csv'

WORDSIM353_SIM = './similarity/wordsim353_sim.csv'
WORDSIM353_REL = './relatedness/wordsim353_rel.csv'

TOP_K = 10


# Evaluate on Analogies dataset. Calculate top1 and top10 accuracy(%) for each category and average accuracies over
# semantic/syntactic/all categories. Assume semantic categories go first and syntactic category names start with 'gram'
def answer_analogy_questions(analogies_path, embeddings, words2ids, top_k):
    all_questions_init = read_sentences_list(analogies_path)
    # lowercase everything
    all_questions_low = [[j.lower() for j in i] for i in all_questions_init]
    # get rid of oov
    all_questions = [q for q in all_questions_low if q[0] == ":" or
                     (q[0] in words2ids and q[1] in words2ids and q[2] in words2ids and q[3] in words2ids)]
    results = []
    group = []
    print('group_name', '1nn%', '10nn%', sep="\t")
    # answer questions, combining them in groups
    for line in all_questions:
        if line[0] == ':':
            if group:  # if group is not empty, evaluate and print results
                results[-1].extend(
                    answer_questions_in_group(group, embeddings, words2ids, top_k))
                print(results[-1][0], '%.2f' % results[-1]
                      [1], '%.2f' % results[-1][2], sep="\t")
                group = []
            group_name = line[1]
            results.append([group_name])
        else:
            group.append(line)
    # handle last group's results
    results[-1].extend(answer_questions_in_group(group, embeddings, words2ids, top_k))
    print(results[-1][0], '%.2f' % results[-1]
          [1], '%.2f' % results[-1][2], sep="\t")
    # print overall results
    results = [r if len(r) > 1 else r + [0., 0.] for r in results]
    n_syntactic = sum(1 for r in results if r[0].startswith('gram'))
    summarize_analogies_results(results, n_syntactic)


# Answer analogy questions in one group
def answer_questions_in_group(questions, embeddings, words2ids, top_k):
    targets = np.ndarray(shape=(len(questions), embeddings.shape[1]), dtype=np.float32)
    for i, q in enumerate(questions):  # [a,b,c]. d = (b - a) + c
        # target embeddings - closest points to question answers
        targets[i, :] = (embeddings[words2ids[q[1]], :] - embeddings[words2ids[q[0]], :]) \
                        + embeddings[words2ids[q[2]], :]
    distances = np.dot(targets, embeddings.T)
    # number of nearest neighbors we are interested in, +3 to account for question words, which we will ignore then
    num_best = top_k + 3
    # partition instead of sorting as it is way faster
    partitioned = np.argpartition(-distances, num_best, axis=1)[:, : num_best]
    # number of correct answers as a 1st nearest neighbor / in a 10-nearest neighbors range
    num_1nn = 0
    num_10nn = 0
    # answer each question - consider first top_k neighbors ignoring question words
    for i, q in enumerate(questions):
        # convert question words to ids
        q_ids = [words2ids[w] for w in q]
        # sort partition based on distances
        p_i = partitioned[i,:]
        d_i = distances[i,:]
        nearest = p_i[np.argsort(-d_i[p_i])]
        # filter out question words and crop up to top_k
        nearest_filtered = [w for w in nearest if w not in q_ids[:3]][:top_k]

        # check for true answer
        if q_ids[3] in nearest_filtered:
            num_10nn += 1
            if q_ids[3] == nearest_filtered[0]:
                num_1nn += 1

    n_quest = len(questions)
    percent_1nn = num_1nn * 100 / n_quest
    percent_10nn = num_10nn * 100 / n_quest
    return percent_1nn, percent_10nn


# Summarize and print results of Analogies evaluation
def summarize_analogies_results(results, len_syn):
    len_sem = len(results) - len_syn
    avg_1 = sum([r[1] for r in results]) / len(results)
    avg_10 = sum([r[2] for r in results]) / len(results)
    avg_sem_1 = sum([r[1] for r in results[:len_sem]]) / \
        len_sem if len_sem > 0 else 0.
    avg_sem_10 = sum([r[2] for r in results[:len_sem]]) / \
        len_sem if len_sem > 0 else 0.
    avg_syn_1 = sum([r[1] for r in results[len_sem:]]) / len_syn if len_syn > 0 else 0.
    avg_syn_10 = sum([r[2] for r in results[len_sem:]]) / len_syn if len_syn > 0 else 0.
    print("Semantic avg 1nn, 10nn accuracy:", '%.2f' % avg_sem_1, '%.2f' % avg_sem_10, sep="\t")
    print("Syntactic avg 1nn, 10nn accuracy:", '%.2f' % avg_syn_1, '%.2f' % avg_syn_10, sep="\t")
    print("Overall avg 1nn, 10nn accuracy:", '%.2f' %
          avg_1, '%.2f' % avg_10, sep="\t")


# human scores versus cosine similarity correlation test, spearman's / pearson's correlation scores
def human_vs_cos_sim_correlation(human_score_path, embeddings, word2ids):
    human_scores = pd.read_csv(human_score_path)
    # lowercase string values
    human_scores = human_scores.applymap(lambda s: s.lower() if type(s) == str else s)
    len_before = len(human_scores)
    # get rid of oov
    human_scores = human_scores[(human_scores.word1.isin(word2ids)) & (human_scores.word2.isin(word2ids))]
    print('initial size, and after removing OOVs', len_before, len(human_scores), sep="\t")
    # get embeddings for first and second words in pairs
    first_word_ids = [word2ids[k] for k in human_scores.iloc[:, 0]]
    second_word_ids = [word2ids[k] for k in human_scores.iloc[:, 1]]
    first_word_embs = embeddings.take(first_word_ids, axis=0)
    second_word_embs = embeddings.take(second_word_ids, axis=0)
    # calculate cosine similarity
    h_product = np.multiply(first_word_embs, second_word_embs)
    cos_similarity = np.sum(h_product, axis=1)
    rho, p = spearmanr(np.column_stack((human_scores.iloc[:, 2], cos_similarity)))
    print("Spearman's rho, p-value:",
          '{:.2f}'.format(rho), '{:.2e}'.format(p), sep="\t")
    # pears_coef, pears_p = pearsonr(human_scores.iloc[:, 2], cos_similarity)
    # print("Pearson's coef, p-value:", '{:.2f}'.format(pears_coef), '{:.2e}'.format(pears_p))
    return rho, p


# pylint: disable=missing-docstring
# Function to draw visualization of distance between embeddings.
def plot_with_labels(low_dim_embs, labels, filename):
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(
            label,
            xy=(x, y),
            xytext=(5, 2),
            textcoords='offset points',
            ha='right',
            va='bottom')
    plt.savefig(filename)


def main(emb_path, lang):
    print('Reading embeddings from file...')
    embeddings, words2ids, ids2words = \
        read_csv_embeddings(emb_path) if emb_path.endswith('.csv') else read_txt_embeddings(emb_path)
    print('1. Answering analogy questions')
    embeddings = normalize_embeddings(embeddings)
    # choosing right file
    if lang == 'tt':
        answer_analogy_questions(QS_FILE_TT, embeddings, words2ids, TOP_K)
    elif lang == 'ru':
        print("Answering semantic questions:")
        answer_analogy_questions(QS_FILE_RU_SEM, embeddings, words2ids, TOP_K)
        print()
        print("Answering synthetic questions:")
        answer_analogy_questions(QS_FILE_RU_SYN, embeddings, words2ids, TOP_K)
        print()
    elif lang == 'en':
        print("Answering google analogy questions:")
        answer_analogy_questions(QS_FILE_EN, embeddings, words2ids, TOP_K)
        print()
        print("Answering BATS questions:")
        answer_analogy_questions(QS_FILE_EN_BATS, embeddings, words2ids, TOP_K)
        print()

    #print('2. Measuring correlation between human scores and cosine similarity')
    if lang == 'tt':
        print('Words SIMILARITY test:')
        human_vs_cos_sim_correlation(SIM_PATH_TT, embeddings, words2ids)
        print('Words RELATEDNESS test:')
        human_vs_cos_sim_correlation(REL_PATH_TT, embeddings, words2ids)
    elif lang == 'ru':
        print('Words SIMILARITY test (Simlex999):')
        human_vs_cos_sim_correlation(SIMLEX999_RU, embeddings, words2ids)
        print()
        print('Words SIMILARITY test (Simlex965):')
        human_vs_cos_sim_correlation(SIMLEX965_RU, embeddings, words2ids)
    elif lang == 'en':
        print('Words SIMILARITY test (Simlex999):')
        human_vs_cos_sim_correlation(SIMLEX999_EN, embeddings, words2ids)
        print('Words SIMILARITY test (Wordsim353_sim):')
        human_vs_cos_sim_correlation(WORDSIM353_SIM, embeddings, words2ids)
        print('Words RELATEDNESS test (Wordsim353_rel):')
        human_vs_cos_sim_correlation(WORDSIM353_REL, embeddings, words2ids)
    return
    print('3. Visualizing the embeddings')
    try:
        # pylint: disable=g-import-not-at-top
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
        plot_only = 500
        low_dim_embs = tsne.fit_transform(embeddings[:plot_only, :])
        example_labels = [ids2words[i] for i in range(plot_only)]
        plot_with_labels(low_dim_embs, example_labels, emb_path[:-4]  + '_tsne.png')
    except ImportError as ex:
        print('Please install sklearn, matplotlib, and scipy to show embeddings.')
        print(ex)


# set up these params (only)
# EMB_PATH = '../glove/dumped/en_300/vectors.txt'
# EMB_PATH = '../fasttext_gensim/dumped/tt_ft_same_as_w2v_params/emb.txt'
# LANG = 'tt'

LANG = sys.argv[1]
EMB_PATH = sys.argv[2]

main(EMB_PATH, LANG)
