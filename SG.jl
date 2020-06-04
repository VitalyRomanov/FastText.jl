includet("LanguageTools.jl")
includet("FastText.jl")

using .LanguageTools
using .FT
using StatsBase
using Printf
using SharedArrays
# using LinearAlgebra

const MAX_SENT_LENGTH = 1000
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
    shared_grads
    shared_moments
    funct
    wPieces
end

struct moments
    in_f_m
    out_f_m
    bucket_f_m
    in_s_m
    out_s_m
    bucket_s_m
    in_iter
    out_iter
    bucket_iter
end

struct ft_params
    in
    out
    buckets
    atomic_in
    atomic_out
    atomic_buckets
end

struct sg_tools
    compute_in!
    activation
    update_grad_in!
    update_grad_b!
    # update_grad_out!
    w2id
    get_buckets
    sample_neg
    in_voc
    discard
end

init_moments(voc_size, n_dims, n_buckets) = begin
    in_f_m = SharedArray{Float32}(n_dims, voc_size)
    out_f_m = SharedArray{Float32}(n_dims, voc_size)
    bucket_f_m = SharedArray{Float32}(n_dims, n_buckets)

    in_f_m .= 0.
    out_f_m .= 0.
    bucket_f_m .= 0.

    in_s_m = SharedArray{Float32}(n_dims, voc_size)
    out_s_m = SharedArray{Float32}(n_dims, voc_size)
    bucket_s_m = SharedArray{Float32}(n_dims, n_buckets)

    in_s_m .= 0.
    out_s_m .= 0.
    bucket_s_m .= 0.

    in_iter = SharedArray{Float32,1}(voc_size)
    out_iter = SharedArray{Float32,1}(voc_size)
    buckets_iter = SharedArray{Float32,1}(n_buckets)

    in_iter .= 1; out_iter .= 1; buckets_iter .= 1

    moments(
        in_f_m,
        out_f_m,
        bucket_f_m,
        in_s_m,
        out_s_m,
        bucket_s_m,
        in_iter,
        out_iter,
        buckets_iter
    )
end

init_shared_params(voc_size, n_dims, n_buckets) = begin
    in_shared = SharedArray{Float32}(n_dims, voc_size)
    out_shared = SharedArray{Float32}(n_dims, voc_size)
    bucket_shared = SharedArray{Float32}(n_dims, n_buckets)

    in_shared .= (rand(n_dims, voc_size) .- 0.5) / n_dims
    out_shared .= 0. #randn(n_dims, voc_size) / n_dims # init these to zero
    bucket_shared .= (rand(n_dims, n_buckets) .- 0.5) / n_dims

    # in_shared .= in_shared ./ sqrt.(sum(in_shared .* in_shared, dims=1))
    # out_shared .= out_shared ./ sqrt.(sum(out_shared .* out_shared, dims=1))
    # bucket_shared .= bucket_shared ./ sqrt.(sum(bucket_shared .* bucket_shared, dims=1))

    atomic_in = SharedArray{Bool}(voc_size)
    atomic_out = SharedArray{Bool}(voc_size)
    atomic_buckets = SharedArray{Bool}(n_buckets)

    atomic_in .= false; atomic_out .= false; atomic_buckets .= false

    ft_params(
        in_shared,
        out_shared,
        bucket_shared,
        atomic_in,
        atomic_out,
        atomic_buckets
    )
end

init_shared_grads(voc_size, n_dims, n_buckets) = begin
    in_shared = SharedArray{Float32}(n_dims, voc_size)
    out_shared = SharedArray{Float32}(n_dims, voc_size)
    bucket_shared = SharedArray{Float32}(n_dims, n_buckets)

    in_shared .= 0.
    out_shared .= 0.
    bucket_shared .= 0.

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

get_tokens!(c, line, token_buffer) = begin
    # subsampling procedure
    # https://arxiv.org/pdf/1310.4546.pdf
    buffer_pos = 1

    global TOK_RE, MAX_SENT_LENGTH
    tokens = eachmatch(TOK_RE, line)

    for t in tokens
        w = t.match
        w_id_final = -1
        if c.funct.in_voc(w)
            w_id = c.vocab.vocab[w]
            # if !(w_id in keys(c.wPieces))
            #     c.wPieces[w_id] = c.funct.get_buckets(w)
            # end
            if c.funct.discard(w)
                continue
            end
            w_id_final = w_id
            # if !c.funct.discard(w)
            #     w_id_final = w_id
            # end
        end
        token_buffer[buffer_pos] = w_id_final
        buffer_pos += 1
        if buffer_pos > MAX_SENT_LENGTH; break; end
    end

    nTokens = buffer_pos - 1
end

# drop_tokens(c, tokens) = begin
#     # subsampling procedure
#     # https://arxiv.org/pdf/1310.4546.pdf
#     buffer = Array{SubString}(undef, length(tokens))
#     buffer_pos = 1
#
#     # create a dictionary that stores pieces
#     # do not store the string tokens
#     # store word ids in a fxed length array to avoid allocations
#
#     # in_voc(c::SGCorpus, w) = w in keys(c.vocab.vocab);
#     # discard_token(c::SGCorpus, w) = rand() < (1 - sqrt(c.vocab.totalWords * c.params.subsampling_parameter / c.vocab.counts[w]))
#     in_voc = (w) -> w in keys(c.vocab.vocab)
#     discard_token = (w) -> rand() < (1 - sqrt(c.vocab.totalWords * c.params.subsampling_parameter / c.vocab.counts[w]))
#
#     UNK_TOKEN = "UNK"
#     for i in 1:length(tokens)
#         if in_voc(tokens[i]) && !discard_token(tokens[i])
#             buffer[buffer_pos] = tokens[i]
#             buffer_pos += 1
#         end
#     end
#     return buffer[1:buffer_pos-1]
# end

_update_grads!(in_grad_flag::SharedArray{Bool,1}, out_grad_flag::SharedArray{Bool,1},
                in_grad::SharedArray{Float32,2}, out_grad::SharedArray{Float32,2},
                in::SharedArray{Float32,2}, out::SharedArray{Float32,2},
                in_id::Int64, out_id::Int64, label::Float32, lr::Float32, lr_factor::Float32,
                n_dims::Int64, act::Float32) = begin

    # w::Float32 = -label .* (1 .- act) .* lr
    w::Float32 = (act - label) .* lr

    in_grad_flag[in_id] = true
    out_grad_flag[out_id] = true

    i = 1
    while i <= n_dims
        in_old = in[i, in_id]
        out_old = out[i, out_id]
        in_grad[i, in_id] += out_old .* w .* lr_factor
        out_grad[i, out_id] += in_old .* w .* lr_factor
        i += 1
    end
end

# _update_grads2!(in_grad_flag::SharedArray{Bool,1}, out_grad_flag::SharedArray{Bool,1},
#                 in_grad::SharedArray{Float32,2}, out_grad::SharedArray{Float32,2},
#                 in::SharedArray{Float32,2}, out::SharedArray{Float32,2}, buffer::Array{Float32,1},
#                 in_id::Int64, out_id::Int64, label::Float32, lr::Float32, lr_factor::Float32,
#                 n_dims::Int64, act::Float32) = begin
#
#     w::Float32 = -label .* (1 .- act) .* lr
#
#     in_grad_flag[in_id] = true
#     out_grad_flag[out_id] = true
#
#     i = 1
#     while i <= n_dims
#         in_old = buffer[i]
#         out_old = out[i, out_id]
#         in_grad[i, in_id] -= out_old .* w .* lr_factor
#         out_grad[i, out_id] -= in_old .* w
#         i += 1
#     end
# end
#
# _update_grads_in!(in_grad_flag::SharedArray{Bool,1},
#                 in_grad::SharedArray{Float32,2},
#                 out::SharedArray{Float32,2},
#                 in_id::Int64, out_id::Int64, label::Float32, lr::Float32, lr_factor::Float32,
#                 n_dims::Int64, act::Float32) = begin
#
#     w::Float32 = -label .* (1 .- act) .* lr
#
#     in_grad_flag[in_id] = true
#
#     i = 1
#     while i <= n_dims
#         in_grad[i, in_id] += out[i, out_id] .* w .* lr_factor
#         i += 1
#     end
# end
#
# _update_grads_out!(out_grad_flag::SharedArray{Bool,1},
#                 out_grad::SharedArray{Float32,2},
#                 buffer::Array{Float32,1},
#                 out_id::Int64, label::Float32, lr::Float32,
#                 n_dims::Int64, act::Float32) = begin
#
#     w::Float32 = -label .* (1 .- act) .* lr
#
#     out_grad_flag[out_id] = true
#
#     i = 1
#     while i <= n_dims
#         out_grad[i, out_id] += buffer[i] .* w
#         i += 1
#     end
# end

_activation(buffer, out, out_id, n_dims) = begin
    a::Float32 = 0.
    i = 1
    while i <= n_dims
        @inbounds a += buffer[i] .* out[i, out_id]
        i += 1
    end
    sigm(a)
    # sigm(dot(buffer, @view out[:, out_id]))
end

"""
combine `in` representation and representations of `buckets` and
store the result into `buffer`
"""
# _compute_in!(buffer, in_, b_, in_id, b_ids, n_dims) = begin
_compute_in!(buffer, in_, b_, in_id, wPieces, n_dims) = begin
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

    # buffer .= @views in_[:, in_id]
    # for b_id in b_ids
    #     buffer .+= @views b_[:, b_id]
    # end
    # buffer .*= factor

    # buffer .= @views (in_[:, in_id] .+ sum(b_[:, b_ids], dims=2)[:]) .* factor
end

apply_grad(params::SharedArray{Float32,2}, id::Int64, update::Array{Float32,1},
            w::Float32, n_dims::Int32) = begin
    d = 1
    while d <= n_dims
        @inbounds params[d, id] -= update[d] * w
        d += 1
    end
end

update_buf(params::SharedArray{Float32,2}, id::Int64, update::Array{Float32,1},
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

    n_buckets = wPieces[1, in_id] #length(buckets)
    # buckets = wPieces[in_id]

    # prepare to iterate buckets
    # bucket_ind = 1


    # consts
    POS_LBL::Float32 = 1; NEG_LBL::Float32 = 0.

    # define window region
    win_pos = max(1, pos-win_size); win_pos_end = min(n_tok, pos+win_size)

    # f.compute_in!(buffer, in_id, buckets)
    f.compute_in!(buffer, in_id)

    # init
    act = 0.


    lr_factor::Float32 = 1. #/ (1 + length(buckets))


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

            update_buf(out_, out_id, buffer_out, g, n_dims)
            # buffer_out .+= @views c.shared_params.out[:, out_id] .* g

            apply_grad(out_, out_id, buffer, g, n_dims)
            neg_ind += 1
        end

        apply_grad(in_, in_id, buffer_out, lr_factor, n_dims)
        bucket_ind = 1
        while bucket_ind <= n_buckets
            @inbounds b_id = wPieces[bucket_ind + 1, in_id]
            # b_id = buckets[bucket_ind]
            apply_grad(buckets_, b_id, buffer_out, lr_factor, n_dims)
            bucket_ind += 1
        end
        win_pos += 1
    end

    # while win_pos <= win_pos_end
    #     out_id = tokens[win_pos]
    #     if win_pos == pos || out_id == -1; win_pos += 1; continue; end
    #
    #     # skip updating gradients if activation is too strong
    #
    #     act = f.activation(buffer, out_id)
    #     # if act > 0.99; processed += 1; win_pos += 1; continue; end
    #     # if act < 0.01; processed += 1; loss += 5.; win_pos += 1; continue; end
    #     loss += -log(act)
    #     processed += 1
    #
    #     # f.update_grad_out!(out_id, buffer, POS_LBL, lr, act)
    #     f.update_grad_in!(in_id, out_id, POS_LBL, lr, lr_factor, act)
    #
    #     while bucket_ind <= n_buckets
    #         b_id = buckets[bucket_ind]
    #         f.update_grad_b!(b_id, out_id, POS_LBL, lr, lr_factor, act)
    #         bucket_ind += 1
    #     end
    #     win_pos += 1
    # end
    #
    # neg_ind = 1
    # bucket_ind = 1
    #
    # while neg_ind <= n_neg
    #     neg_out_id = f.sample_neg()
    #
    #     act = f.activation(buffer, neg_out_id)
    #     # if act < 0.01; processed += 1; neg_ind += 1; continue; end
    #     # if act > 0.99; processed += 1; loss += 5.; neg_ind += 1; continue; end
    #     loss += -log(1-act)
    #     processed += 1
    #
    #     # f.update_grad_out!(neg_out_id, buffer, NEG_LBL, lr, act)
    #     f.update_grad_in!(in_id, neg_out_id, NEG_LBL, lr, lr_factor, act)
    #
    #     while bucket_ind <= n_buckets
    #         b_id = buckets[bucket_ind]
    #         f.update_grad_b!(b_id, neg_out_id, NEG_LBL, lr, lr_factor, act)
    #         bucket_ind += 1
    #     end
    #     neg_ind += 1
    # end
    loss, processed
end

sigm(x) = (1 ./ (1 + exp.(-x)))

init_negative_sampling(v) = begin
    ordered_words = sort(collect(v.vocab), by=x->x[2])
    probs = zeros(length(ordered_words))
    reverseMap = Dict()
    for (w, id) in ordered_words
        probs[id] = v.counts[w] / v.totalWords
        reverseMap[id] = w
    end
    probs .^= 3/4
    probs ./= sum(probs)
    # # # (size) -> map(id -> reverseMap[id], StatsBase.sample(collect(1:length(probs)), StatsBase.Weights(probs), size))
    # indices = collect(1:length(probs))
    # probs_ = StatsBase.Weights(probs)
    # # # (size) -> StatsBase.sample(indices, probs_, size)
    # # # (size) -> map(id -> reverseMap[id], StatsBase.sample(indices, size))
    # () -> StatsBase.sample(indices, probs_)

    factor = 10^abs(floor(minimum(log10.(probs)))-1)
    probs .*= factor
    freqs = convert.(Int64, round.(probs))
    total = sum(freqs)
    lookup = zeros(Int64, total)
    # @show total
    lookup_pos = 1
    for (ind, count) in enumerate(freqs)
        for i in 1:count
            lookup[lookup_pos] = ind
            lookup_pos += 1
        end
    end
    # Base.Sort.searchsortedfirst
    # @show lookup
    () -> begin
        ind = abs(rand(Int64)) % total + 1
        @inbounds lookup[ind]
    end
end


get_scheduler(c) = begin
    scheduler = nothing
    if total_lines > 20000
        scheduler = (iter) -> begin
            middle = total_lines ÷ 2
            frac = abs(iter - middle) / (middle)
            frac * c.params.learning_rate + (1 - frac) * c.params.learning_rate * 10
        end
    else
        scheduler = (iter) -> c.params.learning_rate
    end
    scheduler = (iter) -> c.params.learning_rate
    scheduler
end

# get_context_processor(c::SGCorpus, sample_neg, get_buckets, w2id) = begin
#     shared_params = c.shared_params
#     shared_grads = c.shared_grads
#     win_size = c.params.win_size
#     n_dims = c.params.n_dims
#     buffer = zeros(Float32, n_dims)
#     (tokens, pos, lr) -> process_context(buffer, sample_neg, get_buckets, w2id, shared_params, shared_grads, win_size, lr, n_dims, tokens, pos)
# end

get_context_processor2(c::SGCorpus) =
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

_apply_g!(atomic, par, grad, n_dims, alpha) = begin
    n = length(atomic)

    i = 1
    while i <= n
        if atomic[i] == false; i += 1; continue; end
        d = 1
        while d <= n_dims
            par[d, i] -= grad[d, i] * alpha
            grad[d, i] = 0.
            d += 1
        end
        atomic[i] = false
        i += 1
    end
end


apply_grads(c, alpha) = begin
    _apply_g!(c.shared_grads.atomic_in, c.shared_params.in,
                c.shared_grads.in, c.params.n_dims, alpha)
    _apply_g!(c.shared_grads.atomic_out, c.shared_params.out,
                c.shared_grads.out, c.params.n_dims, alpha)
    _apply_g!(c.shared_grads.atomic_buckets, c.shared_params.buckets,
                c.shared_grads.buckets, c.params.n_dims, alpha)
    # c.shared_params.in .+= c.shared_grads.in
    # c.shared_params.out .+= c.shared_grads.out
    # c.shared_params.buckets .+= c.shared_grads.buckets
    #
    # c.shared_grads.in .= 0.
    # c.shared_grads.out .= 0.
    # c.shared_grads.buckets .= 0.
    #
    # c.shared_grads.atomic_in .= false
    # c.shared_grads.atomic_out .= false
    # c.shared_grads.atomic_buckets .= false
end


check_weights(c) = begin
    act = sum(c.shared_params.in)
    if isnan(act) || isinf(act)
        throw("Weights spoiled")
    end
    act = sum(c.shared_params.out)
    if isnan(act) || isinf(act)
        throw("Weights spoiled")
    end
    act = sum(c.shared_params.buckets)
    if isnan(act) || isinf(act)
        throw("Weights spoiled")
    end
end


SGCorpus(file, vocab; n_dims=300, n_buckets=10000, min_ngram=3, max_ngram=5,
        win_size=5, learning_rate=0.01, neg_samples_per_context=15,
        subsampling_parameter=1e-4, batch_size=1) = begin

    shared_params = init_shared_params(length(vocab), n_dims, n_buckets)
    shared_grads = init_shared_grads(length(vocab), n_dims, n_buckets)
    shared_moments = init_moments(length(vocab), n_dims, n_buckets)

    in_ = shared_params.in
    out_ = shared_params.out
    b_ = shared_params.buckets
    in_g = shared_grads.in
    out_g = shared_grads.out
    b_g = shared_grads.buckets
    in_g_f = shared_grads.atomic_in
    out_g_f = shared_grads.atomic_out
    b_g_f = shared_grads.atomic_buckets
    MAX_SENT_LEN = 10000
    MAX_SUBWORDS = 200

    params = SGParams(n_dims, n_buckets, length(vocab), win_size,
        min_ngram, max_ngram, neg_samples_per_context, subsampling_parameter,
        learning_rate, batch_size, MAX_SENT_LEN, MAX_SUBWORDS)

    wPieces = SharedArray{Int64}(MAX_SUBWORDS+1, length(vocab)) # first item stores count
    populate_subwords!(vocab, wPieces, min_ngram, max_ngram, n_buckets)

    # compute_in! = (buffer, in_id, bucket_ids) -> _compute_in!(buffer,
    #         in_, b_, in_id, bucket_ids, n_dims)
    compute_in! = (buffer, in_id) -> _compute_in!(buffer,
            in_, b_, in_id, wPieces, n_dims)
    activation = (buffer, out_id) -> _activation(buffer, out_, out_id, n_dims)
    # in_grad_u! = (in_id, out_id, label, lr, lr_factor, act) ->
    #     _update_grads_in!(in_g_f, in_g, out_,
    #                     in_id, out_id, label, lr, lr_factor, n_dims, act)
    # b_grad_u! = (b_id, out_id, label, lr, lr_factor, act) ->
    #     _update_grads_in!(b_g_f, b_g, out_,
    #                     b_id, out_id, label, lr, lr_factor, n_dims, act)
    # out_grad_u! = (out_id, buffer, label, lr, act) ->
    #     _update_grads_out!(out_g_f, out_g, buffer,
    #                     out_id, label, lr, n_dims, act)
    in_grad_u! = (in_id, out_id, label, lr, lr_factor, act) ->
        _update_grads!(in_g_f, out_g_f, in_g, out_g, in_, out_,
                        in_id, out_id, label, lr, lr_factor, n_dims, act)
    b_grad_u! = (b_id, out_id, label, lr, lr_factor, act) ->
        _update_grads!(b_g_f, out_g_f, b_g, out_g, b_, out_,
                        b_id, out_id, label, lr, lr_factor, n_dims, act)
    w2id = (w) -> get(vocab.vocab, w, -1)
    get_buckets = (w) -> get_bucket_ids(w, min_ngram, max_ngram, n_buckets)
    neg_sampler = init_negative_sampling(vocab)
    in_voc = (w) -> w in keys(vocab.vocab)
    # discard_token = (w) -> rand() < (1 - sqrt(vocab.totalWords * subsampling_parameter / vocab.counts[w]))
    discard_token = (w) -> begin
        vd = vocab.counts[w] / vocab.totalWords / subsampling_parameter
        !((sqrt(vd) + 1) / vd > rand())
    end


    funct = sg_tools(compute_in!, activation, in_grad_u!, b_grad_u!, #out_grad_u!,
                w2id, get_buckets, neg_sampler, in_voc, discard_token)

    # wPieces = Dict{Int64, Array{Int64,1}}()

    SGCorpus(file, vocab, params, shared_params, shared_grads, shared_moments, funct, wPieces)
end


populate_subwords!(vocab, wPieces, min_ngram, max_ngram, n_buckets) = begin
    max_subwords = size(wPieces)[1] - 1 # first item stores count
    # init
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

    global MAX_SENT_LENGTH
    token_buffer = -ones(Int64, MAX_SENT_LENGTH)

    scheduler = get_scheduler(c)
    # c_proc = get_context_processor(c, sample_neg, get_buckets, w2id)
    c_proc = get_context_processor2(c)

    for epoch in 1:EPOCHS

        seekstart(c.file)

        learning_rate = c.params.learning_rate

        total_processed = 0
        loss = 0.

        # TODO
        # preallocate array of fixed size for tokens

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
                end
            end

            if total_processed > c.params.batch_size
                # apply_grads(c, learning_rate/total_processed)
                # apply_grads_radam(c, learning_rate/total_processed)
                # check_weights(c)
                total_processed = 0
                loss = 0.
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

# (c::SGCorpus)(;total_lines=0) = begin
#
#     global MAX_SENT_LENGTH
#     token_buffer = -ones(Int64, MAX_SENT_LENGTH)
#
#     scheduler = get_scheduler(c)
#     # c_proc = get_context_processor(c, sample_neg, get_buckets, w2id)
#     c_proc = get_context_processor2(c)
#
#     for epoch in 1:EPOCHS
#
#         seekstart(c.file)
#
#         learning_rate = c.params.learning_rate
#
#         total_processed = 0
#         loss = 0.
#
#         # TODO
#         # preallocate array of fixed size for tokens
#
#         start = time_ns()
#         @time for (ind, line) in enumerate(eachline(c.file))
#             # n_tokens = get_tokens!(c, line, token_buffer)
#             # tokens = collect(map((t) -> get(c.vocab.vocab, t, -1), tokenize(line)))
#             # tokens = tokenize(line)
#             # @show 1, token_buffer[1:n_tokens]
#             # @show 2, tokens
#             # if length(tokens) > 1
#             #     l, t = process_tokens(c_proc, tokens, learning_rate)
#             #     loss += l; total_processed += t
#             # end
#             #
#             # if ind % 100 == 0
#             #     passed_seconds, time_left = compute_lapse(start, ind)
#             #     if length(tokens) > 1
#             #         @printf "\rProcessed %d/%d lines, %dm%ds/%dm%ds, loss %.8f lr %.5f\n" ind total_lines passed_seconds÷60 passed_seconds%60 time_left÷60 time_left%60 loss/total_processed learning_rate
#             #     end
#             # end
#             #
#             # if total_processed > c.params.batch_size
#             #     apply_grads(c)
#             #     # check_weights(c)
#             #     total_processed = 0
#             #     loss = 0.
#             # end
#         end
#         println("")
#     end
#
#     in_m = zeros(Float32, c.params.n_dims, c.params.voc_size)
#     out_m = zeros(Float32, c.params.n_dims, c.params.voc_size)
#     bucket_m = zeros(Float32, c.params.n_dims, c.params.n_buckets)
#
#     in_m[:] = c.shared_params.in[:]
#     out_m[:] = c.shared_params.out[:]
#     bucket_m[:] = c.shared_params.buckets[:]
#
#     FastText(in_m, out_m, bucket_m, c.vocab,
#         c.params.min_ngram, c.params.max_ngram)
# end
