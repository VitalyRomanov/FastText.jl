using ArgParse

include("SG.jl")

# FILENAME = "wiki_00"
# EPOCHS = 1
# FILENAME = "test.txt"
# FILENAME = "/Users/LTV/Desktop/AA.txt"
# FILENAME = "/home/ltv/data/local_run/wikipedia/extracted/en_wiki_plain/AA_J.txt"
# FILENAME = "/Volumes/External/datasets/Language/Corpus/en/en_wiki_tiny/wiki_tiny.txt"


s = ArgParseSettings()
@add_arg_table s begin
    "--epochs", "-e"
        help = ""
        arg_type = Int
        default = 1
    "--min_n"
        help = ""
        arg_type = Int
        default = 3
    "--max_n"
        help = ""
        arg_type = Int
        default = 5
    "--buckets", "-b"
        help = ""
        arg_type = Int
        default = 200000
    "--alpha"
        help = ""
        arg_type = Float64
        default = 1e-2
    "--neg", "-n"
        help = ""
        arg_type = Int
        default = 5
    "--voc", "-v"
        help = ""
        arg_type = Int
        default = 200000
    "--min_voc"
        help = ""
        arg_type = Int
        default = 5
    # "--opt2", "-o"
    #     help = "another option with an argument"
    #     arg_type = Int
    #     default = 0
    # "--flag1"
    #     help = "an option without argument, i.e. a flag"
    #     action = :store_true
    "input"
        help = "a positional argument"
        required = true
end
args = parse_args(s)
println(args)

FILENAME = args["input"]
EPOCHS = args["epochs"]

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

    v = prune(v, voc_size, args["min_voc"])
    v, total_lines
end

corpus_file = open(FILENAME)

v, total_lines = learn_voc(corpus_file, args["voc"])

println("Begin training")
c = SGCorpus(corpus_file, v, learning_rate=args["alpha"], n_buckets=args["buckets"],
        neg_samples_per_context=args["neg"], min_ngram=args["min_n"],
        max_ngram=args["max_n"])

println("Training Parameters:")
@show c.params

# using Juno
ft = c(total_lines=total_lines)

save_ft(ft, "en_300")
FT.export_for_tb(ft, "en_300")
FT.export_w2v(ft, "emb.txt")
