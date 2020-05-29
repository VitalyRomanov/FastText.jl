module Vocabulary

    export Vocab, learnVocab!, prune, get_prob

    mutable struct Vocab
        vocab
        counts
        totalWords
    end

    Vocab() = Vocab(Dict{AbstractString, Int64}(), Dict{AbstractString, Int64}(), 0::Integer)

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

    UNK_TOKEN = "UNK"

    prune(v::Vocab, size::Integer, min_count::Integer) = begin
        sorted_words = sort(collect(v.counts), by=x->x[2], rev=true)
        sorted_words = sorted_words[1:min(length(sorted_words), size)]

        vocab = Dict{AbstractString, Int64}()
        counts = Dict{AbstractString, Int64}()
        totalWords = 0
        for (word, count) in sorted_words
            if count >= min_count
                vocab[word] = length(vocab) + 1
                counts[word] = count
                totalWords += count
            end
        end

        # vocab[UNK_TOKEN] = length(vocab) + 1
        # counts[UNK_TOKEN] = v.totalWords - totalWords
        #
        Vocab(vocab, counts, v.totalWords)
        # Vocab(vocab, counts, totalWords)
    end

    get_prob(v::Vocab, word) = get(v.counts, word, 0) / (v.totalWords + 1)

    Base.length(v::Vocab) = length(v.vocab)

    Base.show(io::IO, v::Vocab) = begin
        show_tails = 20
        println(io, "Vocab([")
        sorted_words = sort(collect(v.counts), by=x->x[2], rev=true)
        for (ind, (word, count)) in enumerate(sorted_words)
            if ((ind > show_tails) && (ind < (length(sorted_words) - show_tails))); continue; end
            println(io, "\t$word\t\t\t$count")
            if (ind == show_tails) println(io,""); println(io,"\t..."); println(io,"");end
        end
        println(io, "], totalWords=$(v.totalWords))")
    end
end

# v = Vocab()

# learnVocab!(v, ["1", "2","3","4","5", "5"])

# @show v
# @show get_prob(v, "1")
# @show get_prob(v, "j")

# v = prune(v, 3)
# @show v
