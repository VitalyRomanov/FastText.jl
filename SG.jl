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
    w2id
    get_buckets
    sample_neg
end



init_shared_params(voc_size, n_dims, n_buckets) = begin
    in_shared = SharedArray{Float32}(n_dims, voc_size)
    out_shared = SharedArray{Float32}(n_dims, voc_size)
    bucket_shared = SharedArray{Float32}(n_dims, n_buckets)

    in_shared .= randn(n_dims, voc_size) / n_dims
    out_shared .= randn(n_dims, voc_size) / n_dims
    bucket_shared .= randn(n_dims, n_buckets) / n_dims

    # in_shared .= in_shared ./ sqrt.(sum(in_shared .* in_shared, dims=1))
    # out_shared .= out_shared ./ sqrt.(sum(out_shared .* out_shared, dims=1))
    # bucket_shared .= bucket_shared ./ sqrt.(sum(bucket_shared .* bucket_shared, dims=1))

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

SGCorpus(file,
        vocab;
        n_dims=300,
        n_buckets=10000,
        min_ngram=3,
        max_ngram=5,
        win_size=5,
        learning_rate=0.01,
        neg_samples_per_context=15,
        subsampling_parameter=1e-4,
        batch_size=1) =
            SGCorpus(file, vocab, SGParams(
                n_dims,
                n_buckets,
                length(vocab),
                win_size,
                min_ngram,
                max_ngram,
                neg_samples_per_context,
                subsampling_parameter,
                learning_rate,
                batch_size
            ), init_shared_params(length(vocab), n_dims, n_buckets),
            init_shared_grads(length(vocab), n_dims, n_buckets),
            )

init_negative_sampling(v) = begin
    ordered_words = sort(collect(v.vocab), by=x->x[2])
    probs = zeros(length(ordered_words))
    reverseMap = Dict()
    for (w, id) in ordered_words
        probs[id] = v.counts[w] / v.totalWords
        reverseMap[id] = w
    end
    probs .^= 3/4
    # (size) -> map(id -> reverseMap[id], StatsBase.sample(collect(1:length(probs)), StatsBase.Weights(probs), size))
    indices = collect(1:length(probs))
    probs_ = StatsBase.Weights(probs)
    # (size) -> StatsBase.sample(indices, probs_, size)
    # (size) -> map(id -> reverseMap[id], StatsBase.sample(indices, size))
    () -> StatsBase.sample(indices)
end

# in_voc(c::SGCorpus, w) = w in keys(c.vocab.vocab);

# discard_token(c::SGCorpus, w) = rand() > (1 - sqrt(c.vocab.totalWords * c.params.subsampling_parameter / c.vocab.counts[w]))

# drop_tokens(c, tokens) = begin
#     # subsampling procedure
#     # https://arxiv.org/pdf/1310.4546.pdf
#     buffer = Array{Int32}(undef, length(tokens))
#     buffer_pos = 1
#
#     UNK_TOKEN = "UNK"
#     for i in 1:length(tokens)
#         if in_voc(c, tokens[i]) && discard_token(c, tokens[i])
#             buffer[buffer_pos] = get_w_id(c, tokens[i])
#             buffer_pos += 1
#         end
#     end
#     return buffer[1:buffer_pos-1]
# end

get_w_id(c::SGCorpus, w) = c.vocab.vocab[w]

get_neg_sampler(c) = begin
    neg_sampler = init_negative_sampling(c.vocab)
    # samples_per_context = c.params.neg_samples_per_context
    # sample_neg = () -> neg_sampler(samples_per_context)
    # sample_neg
    neg_sampler
end

get_bucket_fnct(c) = begin
    min_ngram = c.params.min_ngram
    max_ngram = c.params.max_ngram
    max_bucket = c.params.n_buckets
    get_buckets = (w) -> get_bucket_ids(w, min_ngram, max_ngram, max_bucket)
    get_buckets
end

get_vocab_fnct(c) = begin
    v = c.vocab.vocab
    w2id = (w) -> get(v, w, -1) # -1 of oov # did not seem to be beneficial
    w2id
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

get_context_processor(c::SGCorpus, sample_neg, get_buckets, w2id) = begin
    shared_params = c.shared_params
    shared_grads = c.shared_grads
    win_size = c.params.win_size
    n_dims = c.params.n_dims
    buffer = zeros(Float32, n_dims)
    (tokens, pos, lr) -> process_context(buffer, sample_neg, get_buckets, w2id, shared_params, shared_grads, win_size, lr, n_dims, tokens, pos)
end

get_context_processor2(c::SGCorpus, f) = begin
    shared_params = c.shared_params
    shared_grads = c.shared_grads
    win_size = c.params.win_size
    n_dims = c.params.n_dims
    buffer = zeros(Float32, n_dims)
    (tokens, pos, lr) -> _process_context(buffer, f, win_size, lr, tokens, pos)
end

get_in_representation_computer(c::SGCorpus) = begin
    n_dims = c.params.n_dims
    in_ = c.shared_params.in
    b_ = c.shared_params.buckets
    (buffer, in_id, bucket_ids) -> _compute_in!(buffer, in_, b_, in_id, bucket_ids, n_dims)
end

get_activation_computer(c::SGCorpus) = begin
    n_dims = c.params.n_dims
    out_ = c.shared_params.out
    (buffer, out_id) -> _activation(buffer, out_, out_id, n_dims)
end

get_in_grad_updater(c::SGCorpus) = begin
    in_grad = c.shared_grads.in
    out_grad = c.shared_grads.out
    in_ = c.shared_params.in
    out_ = c.shared_params.out
    n_dims = c.params.n_dims
    (in_id, out_id, label, lr, lr_factor, act) ->
    _update_grads!(in_grad, out_grad,
                    in_, out_,
                    in_id, out_id, label, lr, lr_factor,
                    n_dims, act)
end

get_bucket_grad_updater(c::SGCorpus) = begin
    b_grad = c.shared_grads.buckets
    out_grad = c.shared_grads.out
    b_ = c.shared_params.buckets
    out_ = c.shared_params.out
    n_dims = c.params.n_dims
    (b_id, out_id, label, lr, lr_factor, act) ->
    _update_grads!(b_grad, out_grad,
                    b_, out_,
                    b_id, out_id, label, lr, lr_factor,
                    n_dims, act)
end

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


apply_grads(c) = begin
    c.shared_params.in .+= c.shared_grads.in
    c.shared_params.out .+= c.shared_grads.out
    c.shared_params.buckets .+= c.shared_grads.buckets

    c.shared_grads.in .= 0.
    c.shared_grads.out .= 0.
    c.shared_grads.buckets .= 0.
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

    # c.shared_params.in .= c.shared_params.in ./ sqrt.(sum(c.shared_params.in .^ 2, dims=1))
    # c.shared_params.out .= c.shared_params.out ./ sqrt.(sum(c.shared_params.out .^ 2, dims=1))
    # c.shared_params.buckets .= c.shared_params.buckets ./ sqrt.(sum(c.shared_params.buckets .^ 2, dims=1))
end
