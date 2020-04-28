# module SkipGramCorpus
# cd("/Users/LTV/dev/FastText.jl/")
# include("Vocab.jl")
include("LanguageTools.jl")
include("FastText.jl")

using .FT
using SharedArrays

# FILENAME = "wiki_01"
FILENAME = "/Volumes/External/datasets/Language/Corpus/en/en_wiki_tiny/wiki_tiny.txt"

# using .Vocabulary
using .LanguageTools
using StatsBase
using Printf
# using JSON



# export SGCorpus

struct SGParams
    n_dims::Int32
    n_buckets::Int32
    voc_size::Int32
    win_size::Int32
    min_ngram::Int32
    max_ngram::Int32
    neg_samples_per_context::Int32
    subsampling_parameter::Float32
    learning_rate::Float32
end

Base.show(io::IO, params::SGParams) = begin
    println(io, "\n\tn_dims = $(params.n_dims)")
    println(io, "\tvoc_size = $(params.voc_size)")
    println(io, "\tn_buckets = $(params.n_buckets)")
    println(io, "\twin_size = $(params.win_size)")
    println(io, "\tmin_ngram = $(params.min_ngram)")
    println(io, "\tmax_ngram = $(params.max_ngram)")
    println(io, "\tneg_samples_per_context = $(params.neg_samples_per_context)")
    println(io, "\tsubsampling_parameter = $(params.subsampling_parameter)")
    println(io, "\tlearning_rate = $(params.learning_rate)")
end

struct SGCorpus
    file
    vocab
    params
    shared_params
end

struct ft_params
    in
    out
    buckets
    atomic_in
    atomic_out
    atomic_buckets
end



init_shared_params(voc_size, n_dims, n_buckets) = begin
    in_shared = SharedArray{Float32}(n_dims, voc_size)
    out_shared = SharedArray{Float32}(n_dims, voc_size)
    bucket_shared = SharedArray{Float32}(n_dims, n_buckets)

    in_shared[:] = randn(n_dims, voc_size)[:]
    out_shared[:] = randn(n_dims, voc_size)[:]
    bucket_shared[:] = randn(n_dims, n_buckets)[:]

    in_shared .= in_shared .* sum(in_shared .* in_shared, dims=2)
    out_shared .= out_shared .* sum(out_shared .* out_shared, dims=2)
    bucket_shared .= bucket_shared .* sum(bucket_shared .* bucket_shared, dims=2)

    atomic_in = SharedArray{Bool}(voc_size)
    atomic_out = SharedArray{Bool}(voc_size)
    atomic_buckets = SharedArray{Bool}(n_buckets)

    atomic_in .= false
    atomic_out .= false
    atomic_buckets .= false

    ft_params(
        in_shared, 
        out_shared, 
        bucket_shared, 
        atomic_in,
        atomic_out, 
        atomic_buckets
    )
end

# TODO
# implement PMI based word Ngram extraction
# https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf


SGCorpus(file, 
        vocab;
        n_dims=300,
        n_buckets=20000,
        min_ngram=3,
        max_ngram=5,
        win_size=5, 
        learning_rate=0.01,
        neg_samples_per_context=15, 
        subsampling_parameter=1e-4) = 
            SGCorpus(file, vocab, SGParams(
                n_dims,
                n_buckets,
                length(vocab),
                win_size,
                min_ngram,
                max_ngram,
                neg_samples_per_context,
                subsampling_parameter,
                learning_rate
            ), init_shared_params(
                length(vocab), n_dims, n_buckets
            ))

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

# get_negative(c::SGCorpus, size) = begin
#     if c.neg_buffer_pos + size > length(c.neg_buffer)
#         c.neg_buffer[:] = c.neg_sampling(length(c.neg_buffer))
#         c.neg_buffer_pos = 1
#     end
#     neg_samples = c.neg_buffer[1:c.neg_buffer_pos+size]
#     c.neg_buffer_pos += size + 1
#     return neg_samples
# end


process_line(c::SGCorpus, line) = begin
    tokens = tokenize(line)
    tokens = map(w -> if w in keys(c.vocab.vocab); w; else; UNK_TOKEN; end, tokens)
    tokens = filter(w -> rand() < (1 - sqrt(c.vocab.totalWords * c.subsampling_parameter / c.vocab.counts[w])), tokens)
    tokens
end

generate_positive(c::SGCorpus, ch, tokens, pos) = begin
    win_size = c.params.win_size
    for offset in -win_size:win_size
        if offset == 0; continue; end
        if ((pos + offset) < 1 || (pos + offset) > length(tokens)); continue; end
        put!(ch, ((tokens[pos], tokens[pos+offset]), 1.))
        # println("$(tokens[pos])\t$(tokens[pos+offset])\t1")
    end
end

generate_negative(c::SGCorpus, ch, neg_sampler, tokens, pos) = begin
    for neg in neg_sampler(c.params.neg_samples_per_context)
        put!(ch, ((tokens[pos], neg), 0.))
        # println("$(tokens[pos])\t$(neg)\t0")
    end
end

in_voc(c::SGCorpus, w) = w in keys(c.vocab.vocab);

include_token(c::SGCorpus, w) = rand() < (1 - sqrt(c.vocab.totalWords * c.params.subsampling_parameter / c.vocab.counts[w]))

drop_tokens(c, tokens) = begin
    # subsampling procedure 
    # https://arxiv.org/pdf/1310.4546.pdf
    buffer = Array{SubString}(undef, length(tokens))
    buffer_pos = 1

    UNK_TOKEN = "UNK"
    for i in 1:length(tokens)
        # if !(in_voc(c, tokens[i]))
        #     tokens[i] = UNK_TOKEN
        # end
        if in_voc(c, tokens[i]) && include_token(c, tokens[i])
            buffer[buffer_pos] = tokens[i]
            buffer_pos += 1
        end
    end
    return buffer[1:buffer_pos-1]
end

get_w_id(c::SGCorpus, w) = c.vocab.vocab[w]

# sigm(x) = (1 ./ (1 + exp.(-x)))
sigm(x::Float32)::Float32 = begin
    if x > 7.
        0.9933071490757153
    elseif x < -7.
        0.0066928509242848554
    else
        1 ./ (1 + exp.(-x))
    end
end

update_grads!(  in_grad,
                in_grad_id,
                in::SharedArray{Float32,2}, 
                out::SharedArray{Float32,2}, 
                in_id::Int64, 
                out_id::Int64, 
                label::Float32, 
                lr::Float32,
                n_dims::Int32) = begin

    # TODO 
    # test correctness of this function

    act::Float32 = 0.
    i = 1
    while i <= n_dims
        act += in[i, in_id] .* out[i, out_id]
        i += 1
    end
    # println(act)

    if abs(act) > 10.
        return 
    end

    act = sigm(act .* label)

    if isnan(act) || isinf(act)
        throw("Activation became infinite")
    end

    w::Float32 = - label .* (1 .- act) .* lr

    grad_norm::Float32 = 0.

    i = 1
    while i <= n_dims
        in_old = in[i, in_id]
        out_old = out[i, out_id]
        in_grad[i, in_grad_id] += - out_old .* w
        # @inbounds in[i, in_id] = in_old .- out_old .* w
        out[i, out_id] = out_old .- in_old .* w
        # grad_norm += out_old .* w .* out_old .* w
        i += 1
    end
    # println(grad_norm)
end



process_context(sample_neg, 
                get_buckets, 
                w2id, 
                shared_params, 
                win_size::Int32, 
                learning_rate::Float32, 
                n_dims::Int32, 
                tokens, 
                pos::Int64
            ) = begin
    context = tokens[pos]
    in_id = w2id(context)
    buckets = get_buckets(context)

    in_grad = zeros(n_dims, 1)
    buckets_grad = zeros(n_dims, length(buckets))

    # TODO 
    # make parallel
    # maybe need to make parallel at a level of process_context
    # to reduce collisions. but nteed to move a lot of stuff into 
    # workers. neet to restructure all structures...
    
    POS_LBL::Float32 = 1.
    NEG_LBL::Float32 = -1.

    # TODO
    # update in and bucket gradients only after processign the entire window
    # otherwise gradients seem to oscillate
    

    out_ids = [w2id(tokens[offset]) for offset in max(1, pos-win_size):min(length(tokens), pos+win_size)]

    # TODO
    # scale learning rate with the number of buckets

    
    for out_id in out_ids
        update_grads!(in_grad, 1, shared_params.in, shared_params.out, in_id, out_id, POS_LBL, learning_rate, n_dims)

        for (bucket_ind, bucket) in enumerate(buckets)
            update_grads!(buckets_grad, bucket_ind, shared_params.buckets, shared_params.out, bucket, out_id, POS_LBL, learning_rate, n_dims)
        end

    end

    neg = sample_neg()

    for n in neg
        neg_out_id = n
        update_grads!(in_grad, 1, shared_params.in, shared_params.out, in_id, neg_out_id, NEG_LBL, learning_rate, n_dims)

        for (bucket_ind, bucket) in enumerate(buckets)
            update_grads!(buckets_grad, bucket_ind, shared_params.buckets, shared_params.out, bucket, neg_out_id, NEG_LBL, learning_rate, n_dims)
        end
    end

    shared_params.in[:, in_id] -= in_grad

    for (bucket_ind, bucket) in enumerate(buckets)
        shared_params.buckets[:, bucket] -= buckets_grad[:, bucket_ind]
    end
end

process_tokens(c::SGCorpus, sample_neg, get_buckets, w2id, tokens, learning_rate) = begin
    shared_params = c.shared_params
    win_size = c.params.win_size
    # learning_rate = c.params.learning_rate
    n_dims = c.params.n_dims
    lr::Float32 = learning_rate
    for pos in 1:length(tokens)
        process_context(sample_neg, get_buckets, w2id, shared_params, win_size, lr, n_dims, tokens, pos)
    end
end


evaluate(c::SGCorpus, sample_neg, get_buckets, w2id, tokens) = begin
    shared_params = c.shared_params
    win_size = c.params.win_size

    loss::Float32 = 0.

    for pos in 1:length(tokens)
        context = tokens[pos]
        in_id = w2id(context)
        buckets = get_buckets(context)

        POS_LBL::Float32 = 1.
        NEG_LBL::Float32 = -1.
        
        out_ids = [w2id(tokens[offset]) for offset in max(1, pos-win_size):min(length(tokens), pos+win_size)]
        in_vec = shared_params.in[:, in_id] + sum(shared_params.buckets[:, buckets], dims=2)[:]

        for out_id in out_ids
            out_vec = shared_params.out[:, out_id]
            act = in_vec' * out_vec
            if abs(act) > 10.
                continue
            end
            loss += -log(sigm(act))
        end

        neg = sample_neg()

        for n in neg
            out_vec = shared_params.out[:, n]
            act = - in_vec' * out_vec
            if abs(act) > 10.
                continue
            end
            loss += -log(sigm(act))
        end
    end
    loss
end

(c::SGCorpus)(;total_lines=0) = begin
    neg_sampler = init_negative_sampling(c.vocab)
    samples_per_context = c.params.neg_samples_per_context
    sample_neg = () -> neg_sampler(samples_per_context)

    min_ngram = c.params.min_ngram
    max_ngram = c.params.max_ngram
    max_bucket = c.params.n_buckets
    get_buckets = (w) -> get_bucket_ids(w, min_ngram, max_ngram, max_bucket)

    v = c.vocab.vocab
    w2id = (w) -> get(v, w, -1) # -1 of oov # did not seem to be beneficial

    # totalWords = c.vocab.totalWords
    # subsampling_parameter = c.subsampling_parameter
    # counts = c.vocab.counts
    # include_token = (w) -> rand() < (1 - sqrt(totalWords * subsampling_parameter / counts[w])) # did not seem to be beneficial

    seekstart(c.file)

    heldout_line = ""
    for i in 1:3
        heldout_line *= " " * strip(readline(c.file))
    end

    # heldout_tokens = drop_tokens(c, tokenize(heldout_line))
    eval = (tokens) -> evaluate(c, sample_neg, get_buckets, w2id, tokens)

    scheduler = nothing
    if total_lines > 20000
        scheduler = (iter) -> begin
            middle = total_lines ÷ 2
            frac = abs(iter - middle) / (middle)
            frac * c.params.learning_rate + (1 - frac) * c.params.learning_rate * 100
        end
    else
        scheduler = (iter) -> c.params.learning_rate
    end
    scheduler = (iter) -> c.params.learning_rate
    learning_rate = scheduler(1)

    start = time_ns()
    @time for (ind, line) in enumerate(eachline(c.file))
        tokens = drop_tokens(c, tokenize(line))
        if length(tokens) > 1
            process_tokens(c, sample_neg, get_buckets, w2id, tokens, learning_rate)
        end
        if ind % 100 == 0
            lapse = time_ns()
            passed_seconds = (lapse - start) * 1e-9
            if total_lines > 0
                time_left = passed_seconds * (total_lines / ind - 1.)
            else
                time_left = 0.
            end
            # print("\rProcessed $ind/$total_lines lines, ")
            learning_rate = scheduler(ind)
            @printf "\rProcessed %d/%d lines, %dm%ds/%dm%ds, loss %.4f lr %.5f\n" ind total_lines passed_seconds÷60 passed_seconds%60 time_left÷60 time_left%60 eval(tokens)/length(tokens) learning_rate
        end
    end
    println("")

    in_m = zeros(Float32, c.params.n_dims, c.params.voc_size)
    out_m = zeros(Float32, c.params.n_dims, c.params.voc_size)
    bucket_m = zeros(Float32, c.params.n_dims, c.params.n_buckets)

    in_m[:] = c.shared_params.in[:]
    out_m[:] = c.shared_params.out[:]
    bucket_m[:] = c.shared_params.buckets[:]

    FastText(
        in_m,
        out_m,
        bucket_m,
        c.vocab,
        c.params.min_ngram,
        c.params.max_ngram,
    )
end
# end

v = Vocab()

corpus_file = open(FILENAME)

total_lines = 0
print("Learning vocabulary...")
for (ind, line) in enumerate(eachline(corpus_file))
    global total_lines
    tokens = tokenize(line)
    learnVocab!(v, tokens)
    total_lines = ind
end
println("done")

v = prune(v, 5000)

# ft = FastText(v, 300, bucket_size=20000, min_ngram=3, max_ngram=5)

println("Begin training")
c = SGCorpus(corpus_file, v, learning_rate=0.5)

println("Training Parameters:")
@show c.params

ft = c(total_lines=total_lines)

save_ft(ft, "en_300")