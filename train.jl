include("SG.jl")

# FILENAME = "wiki_00"
EPOCHS = 1
# FILENAME = "test.txt"
FILENAME = "/Users/LTV/Desktop/AA.txt"
# FILENAME = "/home/ltv/data/local_run/wikipedia/extracted/en_wiki_plain/AA_J.txt"
# FILENAME = "/Volumes/External/datasets/Language/Corpus/en/en_wiki_tiny/wiki_tiny.txt"

# TODO
# implement PMI based word Ngram extraction
# https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf


process_tokens(c_proc, tokens, n_tokens, learning_rate) = begin
    lr::Float32 = learning_rate #/ c.params.batch_size
    loss = 0.
    processed = 0
    for pos in 1:n_tokens
        l, p = c_proc(tokens, n_tokens, pos, lr)
        loss += l
        processed += p
    end
    loss, processed
end

learn_voc(file, voc_size) = begin

    v = Vocab()

    total_lines = 0
    print("Learning vocabulary...")
    for (ind, line) in enumerate(eachline(corpus_file))
        # global total_lines

        # tokens = tokenize(line)
        if length(v) < voc_size * 10
            global TOK_RE
            tokens = (t.match for t in eachmatch(TOK_RE, line))
            # tokens = tokenize(line)
            learnVocab!(v, tokens, add_new = true)
        # else
        #     learnVocab!(v, tokens, add_new = false)
        end
        total_lines = ind
    end
    println("done")

    v = prune(v, voc_size, 5)
    v, total_lines
end

corpus_file = open(FILENAME)

v, total_lines = learn_voc(corpus_file, 50000)

println("Begin training")
c = SGCorpus(corpus_file, v, learning_rate=1e-2, n_buckets=200000, neg_samples_per_context=2, max_ngram=3)

println("Training Parameters:")
@show c.params

# using Juno
ft = c(total_lines=total_lines)

save_ft(ft, "en_300")
FT.export_for_tb(ft, "en_300")
FT.export_w2v(ft, "emb.txt")
