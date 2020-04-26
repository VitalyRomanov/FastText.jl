cd("/Users/LTV/dev/FastText.jl/")
using Revise
include("FastText.jl")
include("LanguageTools.jl")
include("SkipgramCorpus.jl")
using .LanguageTools
using .FT
using .SkipGramCorpus
# using Flux
using SharedArrays
using Distributed

struct ft_params
    in
    out
    buckets
end

sigm(x) = (1 ./ (1 + exp.(-x)))

# Flux.@functor FastText

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

# loss(x,y) = begin
#     (id_in, buckets, id_out) = x
#     emb_in = ft.in[id_in, :]
#     emb_buckets = ft.bucket[buckets, :]
#     emb_out = ft.out[id_out, :]

#     e_in = emb_in + sum(emb_buckets, dims=1)[:]

#     Flux.logitbinarycrossentropy(e_in' * emb_out, y)
# end
# opt = Descent(0.01)

init_shared_params(ft::FastText) = begin
    in_shared = SharedArray{Float32}(size(ft.in)...)
    out_shared = SharedArray{Float32}(size(ft.out)...)
    bucket_shared = SharedArray{Float32}(size(ft.bucket)...)

    in_shared[:] = ft.in[:]
    out_shared[:] = ft.out[:]
    bucket_shared[:] = ft.bucket[:]
    ft_params(in_shared, out_shared, bucket_shared)
end

init_shared_grads(ft::FastText) = begin
    in_shared = SharedArray{Float32}(size(ft.in)...)
    out_shared = SharedArray{Float32}(size(ft.out)...)
    bucket_shared = SharedArray{Float32}(size(ft.bucket)...)

    in_shared[:] .= 0.
    out_shared[:] .= 0.
    bucket_shared[:] .= 0.
    ft_params(in_shared, out_shared, bucket_shared)
end



place_sample_on_channel!(channel, 
                        # shared_in, 
                        # shared_out, 
                        # shared_bucket, 
                        m::FastText,
                        sample) = begin
    # println(sample)
    (x, y) = sample
    w_in = String(x[1])
    # w_out = String(x[2])
    id_in = m.vocab.vocab[w_in]
    buckets = get_bucket_ids(m, w_in)
    id_out = x[2] #m.vocab.vocab[w_out]
    # println(buckets)
    placed = 0
    put!(channel, (:in, id_in, id_out, y))
    # println("Placed in")
    placed += 1
    for (ind, bucket) in enumerate(buckets)
        put!(channel, (:bucket, bucket, id_out, y))
        placed += 1
        # println("Placed bucket $ind/$(length(buckets))")
    end
    # ((id_in, buckets, id_out), y)
    return placed
end

format_sample(m::FastText, sample) = begin
    (x, y) = sample
    w_in = String(x[1])
    w_out = String(x[2])
    id_in = m.vocab.vocab[w_in]
    buckets = get_bucket_ids(m, w_out)
    id_out = m.vocab.vocab[w_out]
    ((id_in, buckets, id_out), y)
end


compute_grads(sample_channel, grad_channel, shared_params) = begin
    # wait(sample_channel)
    sample = take!(sample_channel)
    where_, in_id, out_id, label = sample
    
    out_ = shared_params.out

    in_ = nothing
    if where_ == :in
        in_ = shared_params.in
    elseif where_ == :bucket
        in_ = shared_params.buckets
    end

    if label == 0.; label = -1.; end

    in_vec = in_[in_id, :]
    out_vec = out_[out_id, :]

    act = sigm((in_vec' * out_vec) .* label)
    grad_in =  - label * (1 - act) .* out_vec
    grad_out = - label * (1 - act) .* in_vec
    put!(grad_channel, (where_, in_id, out_id, grad_in, grad_out))
end

update_params!(shared_params, grad_sample, lr) = begin
    where_, in_id, out_id, in_grad, out_grad = grad_sample

    out_ = shared_params.out

    in_ = nothing
    if where_ == :in
        in_ = shared_params.in
    elseif where_ == :bucket
        in_ = shared_params.buckets
    end

    in_[in_id, :] .+= in_grad .* lr
    out_[out_id, :] .+= out_grad .* lr

end

train(ft::FastText, c::SGCorpus) = begin
    ft_params_s = init_shared_params(ft)
    # ft_params_grad = init_shared_grads(ft)

    learning_rate = 0.1

    batch_size = 128
    batch_count = 0
    overall_placed = 0

    sample_channel = Channel(batch_size * 1000)
    grad_channel = Channel(batch_size * 1000)

    # TODO
    # much faster than with flux, but cannot parallelize
    # read https://docs.julialang.org/en/v1/manual/parallel-computing/#man-shared-arrays-1 
    # and try remotecall_wait for parallelization
    # 
    # exculding sharedarray from calls can potentially increase the execution speed, but it 
    # results in segfault

    println("Begin training")
    for train_sample in c()
        # println(train_sample)
        # continue
        placed = place_sample_on_channel!(sample_channel, ft, train_sample)
        for ind in 1:placed
        # while isready(sample_channel)
            # println("placed $ind/$placed")
            @async compute_grads(sample_channel, grad_channel, ft_params_s)
        end
        batch_count += 1
        overall_placed += placed
        # println("Placed $batch_count tasks, overall $overall_placed")

        if batch_count == batch_size
            # println("All tasks are placed, waiting completion")
            while isready(grad_channel)
                grad_sample = take!(grad_channel)
                # println(grad_sample)
                update_params!(ft_params_s, grad_sample, learning_rate)
            end
            # println("updated")
            batch_count = 0
            overall_placed = 0
        end

    end
end

# TODO 
# returns none when working with higher dimensions
# Flux is ridiculously show
# only one core utilized 
# better to manually implement gradients
# create a parallel gradient computation with Shared Arrays
# https://docs.julialang.org/en/v1/manual/parallel-computing/#Channels-1

# train(ft::FastText, c::SGCorpus) = begin
#     println("Begin training...")
#     batch_size = 128
#     processed = 0
#     samples = []
#     c_loss = 0.
#     for train_sample in c()
#         push!(samples, format_sample(ft, train_sample))
#         if length(samples) == batch_size
#             processed += length(samples)
#             Flux.train!(loss, params(ft), samples, opt)
#             c_loss = sum(map(x_y -> loss(x_y[1],x_y[2]), samples)) / length(samples)
#             samples = []
#             println("Processed: ", processed," current loss: ", c_loss)
#         end
#     end    
# end

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







