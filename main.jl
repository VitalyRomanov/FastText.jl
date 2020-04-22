using Flux

# include("Embedding.jl")

EMB_SIZE = 15
VOC_SIZE = 10

input_index = [5, 5, 5, 5]
output_indices = [1, 2, 3, 5]

data = []
for i in 1:4
    push!(data, ([input_index[i], output_indices[i]], 1.))
end

# in_matrix = Embedding(VOC_SIZE, EMB_SIZE)
# out_matrix = Embedding(VOC_SIZE, EMB_SIZE)

struct FastText
    in
    out
    bucket
end

FastText(voc_s::Integer, dim_s::Integer, bucket_size::Integer) = FastText(rand(voc_s, dim_s), rand(voc_s, dim_s))

Base.getindex(m::FastText, word::String) = begin
    pieces = in_pieces(word)
    bucket_idx = hash_pieces(pieces)
    bucket_emb = sum(m.bucket[bucket_idx, :], dims=1)
    word_ind = word_lookup(word)
    word_emb = m.in[word_ind,:]

    (bucket_emb + word_emb) / (length(bucket_idx) + 1)
end

Flux.@functor FastText

ft = FastText(VOC_SIZE, EMB_SIZE)

loss(x,y) = Flux.logitbinarycrossentropy(ft.in[x[1]]' * ft.out[x[2]], y)

opt = Descent(0.3)

for _ in 1:100
    Flux.train!(loss, params(ft), data, opt)
end







