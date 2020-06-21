module Vocabulary

    export Vocab, learnVocab!, prune, get_prob

    mutable struct Vocab
        vocab
        counts
        totalWords
    end

    Vocab() = Vocab(Dict{AbstractString, Int64}(), Dict{AbstractString, Int64}(), 0::Integer)

    learnVocab!(v::Vocab, tokens; add_new = true) = begin
        for token in tokens
            addWord!(v, token, add_new = add_new)
        end
    end

    addWord!(v::Vocab, word; add_new = true) = begin
        if !(word in keys(v.vocab))
            if add_new
                v.vocab[word] = length(v.vocab) + 1
                v.counts[word] = 1
            end
        else
            v.counts[word] += 1
        end
        v.totalWords += 1
    end

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

        Vocab(vocab, counts, v.totalWords)
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
