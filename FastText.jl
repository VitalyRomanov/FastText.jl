# using Flux



module FT
# using JLD
using JLD2
using HDF5
import Printf

include("Vocab.jl")
using .Vocabulary

export learnVocab!, Vocab, prune,
        save_ft, load_ft,
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
    bucket_emb = sum(m.bucket[:, bucket_idx], dims=2)[:]
    word_ind = m.vocab.vocab[word]
    word_emb = m.in[:, word_ind]

    (bucket_emb + word_emb[:]) / (length(pieces) + 1)
end

Base.getindex(m::FastText, word::SubString) = Base.getindex(m, String(word))

get_vid(m::FastText, word) = m.vocab.vocab[word]
get_bucket_ids(m::FastText, word) = begin
    pieces = in_pieces(word, m.min_ngram, m.max_ngram)
    bucket_idx = hash_piece.(pieces, size(m.bucket)[2])
    bucket_idx
end

get_bucket_ids(word, min_ngram, max_ngram, max_bucket) = begin
    if max_bucket == 0
        return []
    end
    pieces = in_pieces(word, min_ngram, max_ngram)
    bucket_idx = hash_piece.(pieces, max_bucket)
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

hash_piece(x, max_bucket)::Int64 = hash(x) % max_bucket + 1

# Flux.@functor FastText


save_ft(m::FastText, path) = begin
    jldopen(path * ".jld2", "w") do file
        file["in"] = m.in
        file["out"] = m.out
        file["bucket"] = m.bucket
        file["min_ngram"] = m.min_ngram
        file["max_ngram"] = m.max_ngram
        vocab_g = JLD2.Group(file, "Vocab")
        file["Vocab/vocab"] = m.vocab.vocab
        file["Vocab/counts"] = m.vocab.counts
        file["Vocab/totalWords"] = m.vocab.totalWords
    end
end
    # JLD.save(path, "in_m", m.in, 
    #             "out_m", m.out, 
    #             "bucket_m", m.bucket, 
    #             "vocab", m.vocab, 
    #             "min_ngram", m.min_ngram, 
    #             "max_ngram", m.max_ngram)

load_ft(path) = begin
    ft = nothing
    c = jldopen(path, "r") do file
        ft = FastText(
            file["in"],
            file["out"],
            file["bucket"],
            Vocab(file["Vocab/vocab"], file["Vocab/counts"], file["Vocab/totalWords"]),
            file["min_ngram"],
            file["max_ngram"]
        )
    end
    return ft
end
# FastText(
#     JLD.load(path, "in_m"),
#     JLD.load(path, "out_m"),
#     JLD.load(path, "bucket_m"),
#     JLD.load(path, "vocab"),
#     JLD.load(path, "min_ngram"),
#     JLD.load(path, "max_ngram"),
# )

export_w2v(m::FastText, path) = begin
    sink = open(path, "w")
    write(sink, "$(size(m.in)[2]) $(size(m.in)[1])\n")

    n_dims = size(m.in)[1]
    for word in keys(m.vocab)
        word_ind = m.vocab[word]
        write(sink, "$word")
        for i = 1:n_dims
            s = Printf.@sprintf " %.4f" m.in[i, word_ind];
            write(sink, "$s")
        end
        write(sink, "\n")
    end
    close(sink)
end

end
