include("FastText.jl")

using .FT
using StatsBase
using Printf
using SharedArrays

# these parameters are not really meant to be set by users
const MAX_SENT_LEN = 10000  # used to optimize memory usage
const MAX_SUBWORDS = 200    # used to optimize memory usage
const TOK_RE = Regex("[\\w]+|[\\w]+[\\w-]+|[^\\w\\s]")


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
    batch_size::Int32
    max_sent_len::Int32
    max_subwords::Int32 # used only for training
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
    funct
    wPieces
end

struct ft_params
    in
    out
    buckets
    # atomic_in         # were meant to be used for multithreading
    # atomic_out
    # atomic_buckets
end

struct sg_tools
    compute_in!
    activation
    w2id
    get_buckets
    sample_neg
    in_voc
    discard
end

init_shared_params(voc_size, n_dims, n_buckets) = begin
    in_shared = SharedArray{Float32}(n_dims, voc_size)
    out_shared = SharedArray{Float32}(n_dims, voc_size)
    bucket_shared = SharedArray{Float32}(n_dims, n_buckets)

    # adopt initialization like in gensim
    in_shared .= (rand(n_dims, voc_size) .- 0.5) / n_dims
    out_shared .= 0. #randn(n_dims, voc_size) / n_dims # init these to zero
    bucket_shared .= (rand(n_dims, n_buckets) .- 0.5) / n_dims

    # atomic_in = SharedArray{Bool}(voc_size)
    # atomic_out = SharedArray{Bool}(voc_size)
    # atomic_buckets = SharedArray{Bool}(n_buckets)
    #
    # atomic_in .= false; atomic_out .= false; atomic_buckets .= false

    ft_params(in_shared, out_shared, bucket_shared,
        # atomic_in, atomic_out, atomic_buckets
    )
end


get_tokens!(c, line, token_buffer) = begin
    # subsampling procedure
    # https://arxiv.org/pdf/1310.4546.pdf
    buffer_pos = 1

    global TOK_RE, MAX_SENT_LEN
    tokens = eachmatch(TOK_RE, line)

    for t in tokens
        w = t.match
        w_id_final = -1
        if c.funct.in_voc(w)
            w_id = c.vocab.vocab[w]
            # remains of previous implementation that used dictionary
            # instead of preallocated array
            # if !(w_id in keys(c.wPieces))
            #     c.wPieces[w_id] = c.funct.get_buckets(w)
            # end
            if c.funct.discard(w)
                continue
            end
            w_id_final = w_id
        end
        token_buffer[buffer_pos] = w_id_final
        buffer_pos += 1
        if buffer_pos > c.params.max_sent_len; break; end
    end

    nTokens = buffer_pos - 1
end

_activation(buffer, out, out_id, n_dims) = begin
    a::Float32 = 0.
    i = 1
    while i <= n_dims
        @inbounds a += buffer[i] .* out[i, out_id]
        i += 1
    end
    sigm(a)
end

"""
combine `in` representation and representations of `buckets` and
store the result into `buffer`
"""
# _compute_in!(buffer, in_, b_, in_id, b_ids, n_dims) = begin
_compute_in!(buffer, in_, b_, in_id, wPieces, n_dims) = begin
    # https://julialang.org/blog/2019/07/multithreading/
    n_bs = wPieces[1, in_id] # length(b_ids)
    factor = 1. ./ (1. + n_bs)
    i = 1
    while i <= n_dims
        @inbounds buffer[i] = in_[i, in_id]
        b_i = 1
        while b_i <= n_bs
            @inbounds buffer[i] += b_[i, wPieces[b_i + 1, in_id]] #b_[i, b_ids[b_i]]
            b_i += 1
        end
        @inbounds buffer[i] *= factor
        i += 1
    end
end

apply_grad!(params::SharedArray{Float32,2}, id::Int64, update::Array{Float32,1},
            w::Float32, n_dims::Int32) = begin
    d = 1
    while d <= n_dims
        @inbounds params[d, id] -= update[d] * w
        d += 1
    end
end

update_buf!(params::SharedArray{Float32,2}, id::Int64, update::Array{Float32,1},
            w::Float32, n_dims::Int32) = begin
    d = 1
    while d <= n_dims
        @inbounds update[d] += params[d, id] * w
        d += 1
    end
end

_process_context(in_, out_, buckets_, buffer, buffer_out, f, wPieces, win_size,
                    lr, n_neg, tokens, n_tok, pos, n_dims) = begin
    # init
    loss = 0.
    processed = 0
    # current context
    in_id = tokens[pos]
    if in_id == -1; return loss, processed; end

    # buckets = wPieces[in_id]
    n_buckets = wPieces[1, in_id] #length(buckets)

    # consts
    POS_LBL::Float32 = 1; NEG_LBL::Float32 = 0.

    # define window region
    win_pos = max(1, pos-win_size); win_pos_end = min(n_tok, pos+win_size)

    f.compute_in!(buffer, in_id)

    act = 0.

    while win_pos <= win_pos_end
        @inbounds out_id = tokens[win_pos]
        if win_pos == pos || out_id == -1; win_pos += 1; continue; end

        buffer_out .= 0

        neg_ind = 0
        while neg_ind <= n_neg
            if neg_ind == 0
                out_id = tokens[win_pos]
                lbl = POS_LBL
            else
                out_id = f.sample_neg()
                lbl = NEG_LBL
            end

            act = f.activation(buffer, out_id)

            lbl_act = if lbl == POS_LBL; act; else; 1 - act; end
            if lbl_act > 0.99; processed += 1; neg_ind += 1; continue; end
            if lbl_act < 0.01; processed += 1; loss += 5.; neg_ind += 1; continue; end
            loss += - log(lbl_act)
            processed += 1

            g = (act - lbl) * lr

            update_buf!(out_, out_id, buffer_out, g, n_dims)

            apply_grad!(out_, out_id, buffer, g, n_dims)
            neg_ind += 1
        end

        dummy_lr::Float32 = 1.0
        apply_grad!(in_, in_id, buffer_out, dummy_lr, n_dims)
        bucket_ind = 1
        while bucket_ind <= n_buckets
            @inbounds b_id = wPieces[bucket_ind + 1, in_id]
            ## b_id = buckets[bucket_ind]
            apply_grad!(buckets_, b_id, buffer_out, dummy_lr, n_dims)
            bucket_ind += 1
        end
        win_pos += 1
    end

    loss, processed
end

# can switch to precomputed table for better performance
sigm(x) = (1 ./ (1 + exp.(-x)))

bisect_left(arr, val, start_, end_) = begin
    if start_ == end_
        return start_
    end

    middle = (end_ - start_) ÷ 2 + start_
    if val <= arr[middle]
        return bisect_left(arr, val, start_, middle)
    else
        return bisect_left(arr, val, middle + 1, end_)
    end
end

init_negative_sampling_bisect(v) = begin
    ordered_words = sort(collect(v.vocab), by=x->x[2])
    probs = zeros(length(ordered_words))
    reverseMap = Dict()
    for (w, id) in ordered_words
        probs[id] = v.counts[w]
        reverseMap[id] = w
    end
    probs .^= 3/4
    probs ./= sum(probs)

    cumul = zeros(Int64, length(probs))
    acc = 0
    for i in 1:length(probs)
        acc += convert(Int64, floor(probs[i] * 2^32))
        cumul[i] = acc
    end
    total = cumul[end]
    n_words = length(cumul)

    () -> begin
        ind = abs(rand(Int64)) % total + 1
        bisect_left(cumul, ind, 1, n_words)
    end
end

get_scheduler(c; increase_factor=1.) = begin
    # need experiments for best scheduling practices
    scheduler = nothing
    if total_lines > 20000
        scheduler = (iter) -> begin
            middle = total_lines ÷ 2
            frac = abs(iter - middle) / (middle)
            frac * c.params.learning_rate + (1 - frac) * c.params.learning_rate * increase_factor
        end
    else
        scheduler = (iter) -> c.params.learning_rate
    end
    scheduler = (iter) -> c.params.learning_rate
    scheduler
end

get_context_processor(c::SGCorpus) =
    (tokens, n_tok, pos, lr) -> _process_context(
            c.shared_params.in, c.shared_params.out, c.shared_params.buckets,
            zeros(Float32, c.params.n_dims), zeros(Float32, c.params.n_dims),
            c.funct, c.wPieces, c.params.win_size, lr, c.params.neg_samples_per_context,
            tokens, n_tok, pos, c.params.n_dims)


compute_lapse(start, ind) = begin
    lapse = time_ns()
    passed_seconds = (lapse - start) * 1e-9
    if total_lines > 0
        time_left = passed_seconds * (total_lines / ind - 1.)
    else
        time_left = 0.
    end
    passed_seconds, time_left
end


SGCorpus(file, vocab; n_dims=300, n_buckets=10000, min_ngram=3, max_ngram=5,
        win_size=5, learning_rate=0.01, neg_samples_per_context=15,
        subsampling_parameter=1e-4, batch_size=1) = begin

    shared_params = init_shared_params(length(vocab), n_dims, n_buckets)

    in_ = shared_params.in
    out_ = shared_params.out
    b_ = shared_params.buckets

    global MAX_SENT_LEN, MAX_SUBWORDS

    params = SGParams(n_dims, n_buckets, length(vocab), win_size,
        min_ngram, max_ngram, neg_samples_per_context, subsampling_parameter,
        learning_rate, batch_size, MAX_SENT_LEN, MAX_SUBWORDS)

    # wPieces = Dict{Int64, Array{Int64,1}}() # first implementation used dictionary.
    wPieces = SharedArray{Int64}(MAX_SUBWORDS+1, length(vocab)) # first item stores count
    populate_subwords!(vocab, wPieces, min_ngram, max_ngram, n_buckets)

    compute_in! = (buffer, in_id) -> _compute_in!(buffer,
            in_, b_, in_id, wPieces, n_dims)
    activation = (buffer, out_id) -> _activation(buffer, out_, out_id, n_dims)
    w2id = (w) -> get(vocab.vocab, w, -1)
    get_buckets = (w) -> get_bucket_ids(w, min_ngram, max_ngram, n_buckets)
    neg_sampler = init_negative_sampling_bisect(vocab)
    in_voc = (w) -> w in keys(vocab.vocab)
    # discard_token = (w) -> rand() < (1 - sqrt(vocab.totalWords * subsampling_parameter / vocab.counts[w])) # this discarding procedure is suggested in the original word2vec paper
    discard_token = (w) -> begin # this discarding procedure is implemented in gensim (and possibly facebook's fasttext), it is less aggressive in subsampling, more words are available for training
        vd = vocab.counts[w] / vocab.totalWords / subsampling_parameter
        !((sqrt(vd) + 1) / vd > rand())
    end

    funct = sg_tools(compute_in!, activation,
                w2id, get_buckets, neg_sampler, in_voc, discard_token)

    SGCorpus(file, vocab, params, shared_params, funct, wPieces)
end


populate_subwords!(vocab, wPieces, min_ngram, max_ngram, n_buckets) = begin
    max_subwords = size(wPieces)[1] - 1 # first item stores count
    # init to -1
    wPieces .= -1
    for (w, id) in collect(v.vocab)
        buckets = get_bucket_ids(w, min_ngram, max_ngram, n_buckets)

        wPieces[1, id] = min(length(buckets), max_subwords)

        for (ind, b) in enumerate(buckets)
            if ind > max_subwords; break; end
            wPieces[ind + 1, id] = b
        end
    end
end


(c::SGCorpus)(;total_lines=0) = begin

    token_buffer = -ones(Int64, c.params.max_sent_len)

    scheduler = get_scheduler(c)
    c_proc = get_context_processor(c)

    for epoch in 1:EPOCHS

        seekstart(c.file)

        learning_rate = c.params.learning_rate

        total_processed = 0
        loss = 0.

        start = time_ns()
        @time for (ind, line) in enumerate(eachline(c.file))
            n_tokens = get_tokens!(c, line, token_buffer)

            # dummy_lr::Float32 = 1.0
            if n_tokens > 1
                l, t = process_tokens(c_proc, token_buffer, n_tokens, learning_rate)
                loss += l; total_processed += t
            end

            if ind % 100 == 0
                passed_seconds, time_left = compute_lapse(start, ind)
                if n_tokens > 1
                    @printf "\rProcessed %d/%d lines, %dm%ds/%dm%ds, loss %.8f lr %.5f\n" ind total_lines passed_seconds÷60 passed_seconds%60 time_left÷60 time_left%60 loss/total_processed learning_rate
                    total_processed = 0
                    loss = 0.
                end
            end
        end
        println("")
    end

    in_m = zeros(Float32, c.params.n_dims, c.params.voc_size)
    out_m = zeros(Float32, c.params.n_dims, c.params.voc_size)
    bucket_m = zeros(Float32, c.params.n_dims, c.params.n_buckets)

    in_m[:] = c.shared_params.in[:]
    out_m[:] = c.shared_params.out[:]
    bucket_m[:] = c.shared_params.buckets[:]

    FastText(in_m, out_m, bucket_m, c.vocab,
        c.params.min_ngram, c.params.max_ngram)
end
