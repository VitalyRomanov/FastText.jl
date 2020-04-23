cd("/Users/LTV/dev/FastText.jl/")
using Revise
include("FastText.jl")
include("LanguageTools.jl")
include("SkipgramCorpus.jl")
using .LanguageTools
using .FT
using .SkipGramCorpus
using Flux



Flux.@functor FastText

FILENAME = "wiki_01"
corpus_file = open(FILENAME)

v = Vocab()

print("Learning vocabulary...")
for line in eachline(corpus_file)
    tokens = tokenize(line)
    learnVocab!(v, tokens)
end
v = prune(v, 50000)
println("done")

c = SGCorpus(corpus_file, v)

ft = FastText(v, 300, bucket_size=20000, min_ngram=3, max_ngram=5)
# logitbinarycrossentropy(ŷ, y) = (1 - y).*ŷ - logσ.(ŷ)

loss(x,y) = begin
    (id_in, buckets, id_out) = x
    emb_in = ft.in[id_in, :]
    emb_buckets = ft.bucket[buckets, :]
    emb_out = ft.out[id_out, :]

    e_in = emb_in + sum(emb_buckets, dims=1)[:]

    Flux.logitbinarycrossentropy(e_in' * emb_out, y)
end
opt = Descent(0.01)

format_sample(m::FastText, sample) = begin
    (x, y) = sample
    w_in = String(x[1])
    w_out = String(x[2])
    id_in = m.vocab.vocab[w_in]
    buckets = get_bucket_ids(m, w_out)
    id_out = m.vocab.vocab[w_out]
    ((id_in, buckets, id_out), y)
end

# TODO 
# returns none when working with higher dimensions
# Flux is ridiculously show
# only one core utilized 
# better to manually implement gradients
# create a parallel gradient computation with Shared Arrays

train(ft::FastText, c::SGCorpus) = begin
    println("Begin training...")
    batch_size = 128
    processed = 0
    samples = []
    c_loss = 0.
    for train_sample in c()
        push!(samples, format_sample(ft, train_sample))
        if length(samples) == batch_size
            processed += length(samples)
            Flux.train!(loss, params(ft), samples, opt)
            c_loss = sum(map(x_y -> loss(x_y[1],x_y[2]), samples)) / length(samples)
            samples = []
            println("Processed: ", processed," current loss: ", c_loss)
        end
    end    
end

train(ft, c)


# @show ft["Schopenhauer"]


# EMB_SIZE = 15
# VOC_SIZE = 10
# BUCKET_SIZE = 10

# input_index = [5, 5, 5, 5]
# output_indices = [1, 2, 3, 5]

# data = []
# for i in 1:4
#     push!(data, ([input_index[i], output_indices[i]], 1.))
# end

# ft = FastText(VOC_SIZE, EMB_SIZE, BUCKET_SIZE)

# loss(x,y) = Flux.logitbinarycrossentropy(ft.in[x[1]]' * ft.out[x[2]], y)

# opt = Descent(0.3)

# for _ in 1:100
#     Flux.train!(loss, params(ft), data, opt)
# end







