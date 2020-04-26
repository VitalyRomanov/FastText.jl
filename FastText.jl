# using Flux



module FT

include("Vocab.jl")
using .Vocabulary

export learnVocab!, Vocab, prune,
        FastText, get_bucket_ids

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
        rand(dim_s, length(vocab)), 
        rand(dim_s, length(vocab)), 
        rand(dim_s, bucket_size),
        vocab,
        min_ngram,
        max_ngram
    )

Base.getindex(m::FastText, word) = begin
    pieces = in_pieces(word, m.min_ngram, m.max_ngram)
    bucket_idx = hash_piece.(pieces, length(m.vocab))
    bucket_emb = sum(m.bucket[bucket_idx, :], dims=1)[:]
    word_ind = m.vocab.vocab[word]
    word_emb = m.in[word_ind,:]

    (bucket_emb + word_emb[:]) / (length(pieces) + 1)
end

Base.getindex(m::FastText, word::SubString) = Base.getindex(m, String(word))

get_vid(m::FastText, word) = m.vocab.vocab[word]
get_bucket_ids(m::FastText, word) = begin
    pieces = in_pieces(word, m.min_ngram, m.max_ngram)
    bucket_idx = hash_piece.(pieces, size(m.bucket)[1])
    bucket_idx
end

W_BEGIN = "<"
W_END = ">"

in_pieces(word, min_ngram::Integer, max_ngram::Integer) = begin
    word = W_BEGIN * word * W_END
    pieces = []
    for pos in 1:(length(word)-min_ngram), w in min_ngram:max_ngram
        if !isvalid(word, pos); continue; end
        if pos+w <= length(word)
            if !isvalid(word, pos+w); continue; end
            push!(pieces, word[pos:(pos+w)])
        end
    end
    pieces
end

hash_piece(x, voc_size)::Int64 = hash(x) % voc_size + 1

# Flux.@functor FastText

end
