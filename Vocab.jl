module Vocabulary

    export Vocab, learnVocab!, prune, get_prob

    mutable struct Vocab
        vocab
        counts
        totalWords
    end

    Vocab() = Vocab(Dict(), Dict(), 0::Integer)

    learnVocab!(v::Vocab, tokens) = begin
        for token in tokens
            addWord!(v, token)
        end
    end

    addWord!(v::Vocab, word) = begin
        if !(word in keys(v.vocab))
            v.vocab[word] = length(v.vocab) + 1
            v.counts[word] = 1
        else
            v.counts[word] += 1
        end
        v.totalWords += 1
    end

    prune(v::Vocab, size::Integer) = begin
        sorted_words = sort(collect(v.counts), by=x->x[2], rev=true)
        sorted_words = sorted_words[1:min(length(sorted_words), size)]
        
        vocab = Dict()
        counts = Dict()
        totalWords = 0
        for (word, count) in sorted_words
            vocab[word] = length(vocab) + 1
            counts[word] = count
            totalWords += 1
        end

        Vocab(vocab, counts, totalWords)
    end

    get_prob(v::Vocab, word) = get(v.counts, word, 0) / (v.totalWords + 1)

    Base.length(v::Vocab) = length(v.vocab)
end

# v = Vocab()

# learnVocab!(v, ["1", "2","3","4","5", "5"])

# @show v
# @show get_prob(v, "1")
# @show get_prob(v, "j")

# v = prune(v, 3)
# @show v