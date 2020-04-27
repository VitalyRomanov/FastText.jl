# module SkipGramCorpus
# cd("/Users/LTV/dev/FastText.jl/")
# include("Vocab.jl")
include("LanguageTools.jl")
include("FastText.jl")

using .FT
using SharedArrays

FILENAME = "wiki_01"

# using .Vocabulary
using .LanguageTools
using StatsBase

# export SGCorpus

mutable struct SGCorpus
    file
    vocab
    win_size
    contexts_in_batch
    neg_samples_per_context
    subsampling_parameter
    fasttext
    shared_params
    shared_grad
    neg_sampling
    neg_buffer
    neg_buffer_pos
    learning_rate::Float32
end

struct ft_params
    in
    out
    buckets
    atomic_in
    atomic_out
    atomic_buckets
end


init_shared_params(ft::FastText) = begin
    in_shared = SharedArray{Float32}(size(ft.in)...)
    out_shared = SharedArray{Float32}(size(ft.out)...)
    bucket_shared = SharedArray{Float32}(size(ft.bucket)...)

    in_shared[:] = ft.in[:]
    out_shared[:] = ft.out[:]
    bucket_shared[:] = ft.bucket[:]

    atomic_in = SharedArray{Bool}(size(ft.in)[1])
    atomic_out = SharedArray{Bool}(size(ft.out)[1])
    atomic_buckets = SharedArray{Bool}(size(ft.bucket)[1])

    atomic_in .= false
    atomic_out .= false
    atomic_buckets .= false

    ft_params(in_shared, out_shared, bucket_shared, atomic_in, atomic_out, atomic_buckets)
end

init_shared_grads(ft::FastText) = begin
    in_shared = SharedArray{Float32}(size(ft.in)...)
    out_shared = SharedArray{Float32}(size(ft.out)...)
    bucket_shared = SharedArray{Float32}(size(ft.bucket)...)

    in_shared[:] .= 0.
    out_shared[:] .= 0.
    bucket_shared[:] .= 0.

    atomic_in = SharedArray{Bool}(size(ft.in)[1])
    atomic_out = SharedArray{Bool}(size(ft.out)[1])
    atomic_buckets = SharedArray{Bool}(size(ft.bucket)[1])

    atomic_in .= false
    atomic_out .= false
    atomic_buckets .= false

    ft_params(in_shared, out_shared, bucket_shared, atomic_in, atomic_out, atomic_buckets)
end

# TODO
# implement PMI based word Ngram extraction
# https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf

SGCorpus(file, vocab, fasttext, lr; win_size=5, contexts_in_batch=15, neg_samples_per_context=15, subsampling_parameter=1e-4) = 
SGCorpus(file, vocab, win_size, contexts_in_batch, neg_samples_per_context, subsampling_parameter, fasttext, init_shared_params(fasttext), init_shared_grads(fasttext), init_negative_sampling(vocab), zeros(Integer, 100000), 100000, lr)

init_negative_sampling(v) = begin
    ordered_words = sort(collect(v.vocab), by=x->x[2])
    probs = zeros(length(ordered_words))
    reverseMap = Dict()
    for (w, id) in ordered_words
        probs[id] = v.counts[w] / v.totalWords
        reverseMap[id] = w
    end
    probs .^= 3/4
    # TODO
    # need only ids, remove work lookup
    # (size) -> map(id -> reverseMap[id], StatsBase.sample(collect(1:length(probs)), StatsBase.Weights(probs), size))
    indices = collect(1:length(probs))
    probs_ = StatsBase.Weights(probs)
    (size) -> StatsBase.sample(indices, probs_, size)
end

get_negative(c::SGCorpus, size) = begin
    if c.neg_buffer_pos + size > length(c.neg_buffer)
        c.neg_buffer[:] = c.neg_sampling(length(c.neg_buffer))
        c.neg_buffer_pos = 1
    end
    neg_samples = c.neg_buffer[1:c.neg_buffer_pos+size]
    c.neg_buffer_pos += size + 1
    return neg_samples
end


process_line(c::SGCorpus, line) = begin
    tokens = tokenize(line)
    tokens = map(w -> if w in keys(c.vocab.vocab); w; else; UNK_TOKEN; end, tokens)
    tokens = filter(w -> rand() < (1 - sqrt(c.vocab.totalWords * c.subsampling_parameter / c.vocab.counts[w])), tokens)
    tokens
end

generate_positive(c::SGCorpus, ch, tokens, pos) = begin
    for offset in -c.win_size:c.win_size
        if offset == 0; continue; end
        if ((pos + offset) < 1 || (pos + offset) > length(tokens)); continue; end
        put!(ch, ((tokens[pos], tokens[pos+offset]), 1.))
        # println("$(tokens[pos])\t$(tokens[pos+offset])\t1")
    end
end

generate_negative(c::SGCorpus, ch, neg_sampler, tokens, pos) = begin
    for neg in neg_sampler(c.neg_samples_per_context)
        put!(ch, ((tokens[pos], neg), 0.))
        # println("$(tokens[pos])\t$(neg)\t0")
    end
end

in_voc(c::SGCorpus, w) = w in keys(c.vocab.vocab);

include_token(c::SGCorpus, w) = rand() < (1 - sqrt(c.vocab.totalWords * c.subsampling_parameter / c.vocab.counts[w]))

drop_tokens(c::SGCorpus, tokens) = begin
    # subsampling procedure 
    # https://arxiv.org/pdf/1310.4546.pdf
    buffer = Array{SubString}(undef, length(tokens))
    buffer_pos = 1

    UNK_TOKEN = "UNK"
    for i in 1:length(tokens)
        if !(in_voc(c, tokens[i]))
            tokens[i] = UNK_TOKEN
        end
        if include_token(c, tokens[i])
            buffer[buffer_pos] = tokens[i]
            buffer_pos += 1
        end
    end
    return buffer[1:buffer_pos-1]
end

get_w_id(c::SGCorpus, w) = c.vocab.vocab[w]

sigm(x) = (1 ./ (1 + exp.(-x)))

update_grads!(in_grad::SharedArray{Float32,2}, 
                out_grad::SharedArray{Float32,2}, 
                in::SharedArray{Float32,2}, 
                out::SharedArray{Float32,2}, 
                in_id::Int64, 
                out_id::Int64, 
                label::Float32, 
                lr::Float32) = begin

    # while (atomic_in[in_id] || atomic_out[out_id])
    #     sleep(0.00001)
    # end

    # atomic_in[in_id] = true
    # atomic_out[out_id] = true

    # TODO
    # flexible dimensionality

    act::Float32 = 0.
    i = 1
    while i <= 300
        @inbounds act += in[i, in_id] .* out[i, out_id]
        i += 1
    end
    act = sigm(act .* label)

    w::Float32 = - label .* (1 .- act) .* lr

    i = 1
    while i <= 300
        @inbounds in_old = in[i, in_id]
        @inbounds out_old = out[i, out_id]
        @inbounds in[i, in_id] = in_old .+ out_old .* w
        @inbounds out[i, out_id] = out_old .+ in_old .* w
        i += 1
    end


    # @inbounds in_grad[:, in_id] .+= @views in[:, in_id] .+ out[:, out_id] .* w
    # @inbounds out_grad[:, out_id] .+= @views out[:, out_id] .+ in[:, in_id] .* w

    # atomic_in[in_id] = false
    # atomic_out[out_id] = false

    # # println("Views")
    # in_vec = in[in_id, :]
    # out_vec = out[out_id, :]
    # # println("Activation")
    # act = sigm((in_vec' * out_vec) .* label)

    # in_vec .*= - label .* (1 .- act)
    # out_vec .*= - label .* (1 .- act)

    # # println("Update")
    # in[in_id, :] .+= out_vec
    # out[out_id, :] .+= in_vec

    # grad_in =  - label * inv_act .* out_vec
    # grad_out = - label * inv_act .* in_vec
    # grad_in, grad_out
end



process_context(c::SGCorpus, sample_neg, tokens, pos) = begin
    context = tokens[pos]
    in_id = get_w_id(c, context)
    buckets = get_bucket_ids(c.fasttext, context)

    # TODO 
    # make parallel
    # maybe need to make parallel at a level of process_context
    # to reduce collisions. but nteed to move a lot of stuff into 
    # workers. neet to restructure all structures...
    
    POS_LBL::Float32 = 1.
    NEG_LBL::Float32 = -1.

    out_ids = [get_w_id(c, tokens[offset]) for offset in max(1, pos-c.win_size):min(length(tokens), pos+c.win_size)]

    
    for out_id in out_ids
        update_grads!(c.shared_grad.in, c.shared_grad.out, c.shared_params.in, c.shared_params.out, in_id, out_id, POS_LBL, c.learning_rate)

        for bucket in buckets
            update_grads!(c.shared_grad.buckets, c.shared_grad.out, c.shared_params.buckets, c.shared_params.out, bucket, out_id, POS_LBL, c.learning_rate)
        end

    end

    neg = c.neg_sampling(c.neg_samples_per_context)
    # @time neg = get_negative(c, c.neg_samples_per_context)
    for n in neg
        neg_out_id = n
        c.shared_grad.in, c.shared_grad.out
        update_grads!(c.shared_grad.in, c.shared_grad.out, c.shared_params.in, c.shared_params.out, in_id, neg_out_id, NEG_LBL, c.learning_rate)

        for bucket in buckets
            update_grads!(c.shared_grad.buckets, c.shared_grad.out, c.shared_params.buckets, c.shared_params.out, bucket, neg_out_id, NEG_LBL, c.learning_rate)
        end
    end
    # for _ in 1:c.neg_samples_per_context
    #     neg_out_id = c.neg_sampling(1)
    # end
end

process_tokens(c::SGCorpus, sample_neg, tokens) = begin
    for pos in 1:length(tokens)
        process_context(c, sample_neg, tokens, pos)
    end
end

(c::SGCorpus)() = begin
    neg_sampler = init_negative_sampling(c.vocab)
    samples_per_context = c.neg_samples_per_context
    sample_neg = () -> neg_sampler(samples_per_context)

    

    seekstart(c.file)
    @time for (ind, line) in enumerate(eachline(c.file))
        tokens = drop_tokens(c, tokenize(line))
        if length(tokens) > 1
            process_tokens(c, sample_neg, tokens)
        end
        println("Processed $ind lines")
    end
end
# end

v = Vocab()

corpus_file = open(FILENAME)

for line in eachline(corpus_file)
    tokens = tokenize(line)
    learnVocab!(v, tokens)
end

v = prune(v, 1000)

ft = FastText(v, 300, bucket_size=20000, min_ngram=3, max_ngram=5)

lr = 0.01

c = SGCorpus(corpus_file, v, ft, lr)

c()
# for item in chnl
#     @show item
# end