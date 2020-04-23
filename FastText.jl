# using Flux



module FT

include("Vocab.jl")
using .Vocabulary

export learnVocab!, Vocab, prune,
        FastText

struct FastText
    in
    out
    bucket
    vocab
    min_ngram
    max_ngram
end

# FastText(voc_s::Integer, dim_s::Integer, vocab::Vocab; bucket_size::Integer=2000) = 
#     FastText(
#         rand(voc_s, dim_s), 
#         rand(voc_s, dim_s), 
#         rand(bucket_size, dim_s),
#         vocab
#     )

FastText(vocab::Vocab, 
        dim_s::Int64;
        bucket_size::Int64=20000, 
        min_ngram::Int64=3, 
        max_ngram::Int64=5) = 
    FastText(
        rand(length(vocab), dim_s), 
        rand(length(vocab), dim_s), 
        rand(bucket_size, dim_s),
        vocab,
        min_ngram,
        max_ngram
    )

Base.getindex(m::FastText, word::String) = begin
    pieces = in_pieces(word, m.min_ngram, m.max_ngram)
    bucket_idx = hash_piece.(pieces, length(m.vocab))
    bucket_emb = sum(m.bucket[bucket_idx, :], dims=1)[:]
    word_ind = m.vocab.vocab[word]
    word_emb = m.in[word_ind,:]

    (bucket_emb + word_emb[:]) / (length(pieces) + 1)
end

W_BEGIN = "<"
W_END = ">"

in_pieces(word::String, min_ngram::Integer, max_ngram::Integer) = begin
    word = W_BEGIN * word * W_END
    pieces = []
    for pos in 1:(length(word)-min_ngram), w in min_ngram:max_ngram
        if pos+w <= length(word)
            push!(pieces, word[pos:(pos+w)])
        end
    end
    pieces
end

hash_piece(x, voc_size) = hash(x) % voc_size

# Flux.@functor FastText

end
