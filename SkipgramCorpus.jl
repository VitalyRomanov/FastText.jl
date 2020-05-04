module SkipGramCorpus
# cd("/Users/LTV/dev/FastText.jl/")
# include("Vocab.jl")
include("LanguageTools.jl")

# FILENAME = "wiki_01"

# using .Vocabulary
using .LanguageTools
using StatsBase

export SGCorpus

struct SGCorpus
    file
    vocab
    win_size
    contexts_in_batch
    neg_samples_per_context
    subsampling_parameter
end

# TODO
# implement PMI based word Ngram extraction
# https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf

SGCorpus(file, vocab; win_size=5, contexts_in_batch=15, neg_samples_per_context=15, subsampling_parameter=1e-4) = 
SGCorpus(file, vocab, win_size, contexts_in_batch, neg_samples_per_context, subsampling_parameter)

init_negative_sampling(c::SGCorpus) = begin
    ordered_words = sort(collect(c.vocab.vocab), by=x->x[2])
    probs = zeros(length(ordered_words))
    reverseMap = Dict()
    for (w, id) in ordered_words
        probs[id] = c.vocab.counts[w] / c.vocab.totalWords
        reverseMap[id] = w
    end
    probs .^= 3/4
    # TODO
    # need only ids, remove work lookup
    # (size) -> map(id -> reverseMap[id], StatsBase.sample(collect(1:length(probs)), StatsBase.Weights(probs), size))
    indices = collect(1:length(probs))
    probs_ = StatsBase.Weights(probs)
    (size) -> map(id -> reverseMap[id], StatsBase.sample(indices, probs_, size))
end

UNK_TOKEN = "UNK"

process_line(c::SGCorpus, line) = begin
    tokens = tokenize(line)
    tokens = map(w -> if w in keys(c.vocab.vocab); w; else; UNK_TOKEN; end, tokens)
    tokens = filter(w -> rand() < (1 - sqrt(c.vocab.totalWords * c.subsampling_parameter / c.vocab.counts[w])), tokens)
    tokens
end

generate_positive(c::SGCorpus, ch, tokens, pos) = begin
    for offset in -c.win_size:c.win_size
        if offset == 0; continue; end
        if ((pos + offset) < 1 || (pos + offset) > length(tokens)); continue; end
        put!(ch, ((tokens[pos], tokens[pos+offset]), 1.))
        # println("$(tokens[pos])\t$(tokens[pos+offset])\t1")
    end
end

generate_negative(c::SGCorpus, ch, neg_sampler, tokens, pos) = begin
    for neg in neg_sampler(c.neg_samples_per_context)
        put!(ch, ((tokens[pos], neg), 0.))
        # println("$(tokens[pos])\t$(neg)\t0")
    end
end

(c::SGCorpus)() = begin
    neg_sampler = init_negative_sampling(c)

    # TODO 
    # this procedure generates 300 w/s. It is impossible to be faster than Gensim at this rate


    chnl = Channel() do ch
        seekstart(c.file)
        for (ind, line) in enumerate(eachline(c.file))
            # subsampling procedure 
            # https://arxiv.org/pdf/1310.4546.pdf
            # tokens = tokenize(line)
            # tokens = map(w -> if w in keys(c.vocab.vocab); w; else; UNK_TOKEN; end, tokens)
            # tokens = filter(w -> rand() < (1 - sqrt(c.vocab.totalWords * c.subsampling_parameter / c.vocab.counts[w])), tokens)
            # println("Tokens")
            tokens = process_line(c, line)
            if length(tokens) > 1
                for pos in 1:length(tokens)
                    # println("Pos")
                    generate_positive(c, ch, tokens, pos)
                    # println("Neg")
                    generate_negative(c, ch, neg_sampler, tokens, pos)
                end
            end
            println("Processed $ind lines")
        end
    end
    return chnl
end
end

# v = Vocab()

# corpus_file = open(FILENAME)

# for line in eachline(corpus_file)
#     tokens = tokenize(line)
#     learnVocab!(v, tokens)
# end

# v = prune(v, 1000)

# c = SGCorpus(corpus_file, v)

# chnl = c()
# for item in chnl
#     @show item
# end