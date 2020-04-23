using Flux

include("Vocab.jl")
using .Vocabulary


struct FastText
    in
    out
    bucket
    vocab
end

FastText(voc_s::Integer, dim_s::Integer, vocab::Vocab; bucket_size::Integer=2000) = 
    FastText(
        rand(voc_s, dim_s), 
        rand(voc_s, dim_s), 
        rand(bucket_size, dim_s),
        vocab
    )

FastText(vocab::Vocab, dim_s::Integer; bucket_size::Integer=2000) = 
    FastText(
        rand(length(vocab), dim_s), 
        rand(length(vocab), dim_s), 
        rand(bucket_size, dim_s),
        vocab
    )

Base.getindex(m::FastText, word::String) = begin
    pieces = in_pieces(word)
    bucket_idx = hash_pieces(pieces)
    bucket_emb = sum(m.bucket[bucket_idx, :], dims=1)
    word_ind = word_lookup(word)
    word_emb = m.in[word_ind,:]

    (bucket_emb + word_emb) / (length(bucket_idx) + 1)
end

Flux.@functor FastText


