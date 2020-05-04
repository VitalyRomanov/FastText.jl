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
        neg_samples_per_context=500, 
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
    (size) -> StatsBase.sample(indices, probs_, size)
end