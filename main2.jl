# module SkipGramCorpus
# cd("/Users/LTV/dev/FastText.jl/")
# include("Vocab.jl")
include("LanguageTools.jl")
include("FastText.jl")
include("SG.jl")

using .FT
using SharedArrays

# FILENAME = "wiki_01"
EPOCHS = 1
# FILENAME = "test.txt"
FILENAME = "/Users/LTV/Desktop/AA_t.txt"
# FILENAME = "/Volumes/External/datasets/Language/Corpus/en/en_wiki_tiny/wiki_tiny.txt"

# using .Vocabulary
using .LanguageTools
using StatsBase
using Printf
# using JSON

# using Distributed
# addprocs(2)
# julia -p <n> -L file1.jl -L file2.jl driver.jl
# r = @spawnat :any rand(2,2)
# s = @spawnat :any 1 .+ fetch(r)
# the act of shipping the closure ()->sum(A) to worker 2 results in Main.A being defined on 2. Main.A continues to exist on worker 2 even after the call remotecall_fetch returns
# S = SharedArray{Int,2}()
# @sync begin
#     for p in procs(S)
#         @async begin
#             remotecall_wait(fill!, p, S, p)
#         end
#     end
# end

# function advection_shared!(q, u)
#     @sync begin
#         for p in procs(q)
#             @async remotecall_wait(advection_shared_chunk!, :any, q, u)
#         end
#     end
#     q
# end;


# export SGCorpus



# TODO
# implement PMI based word Ngram extraction
# https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf


# get_negative(c::SGCorpus, size) = begin
#     if c.neg_buffer_pos + size > length(c.neg_buffer)
#         c.neg_buffer[:] = c.neg_sampling(length(c.neg_buffer))
#         c.neg_buffer_pos = 1
#     end
#     neg_samples = c.neg_buffer[1:c.neg_buffer_pos+size]
#     c.neg_buffer_pos += size + 1
#     return neg_samples
# end


# process_line(c::SGCorpus, line) = begin
#     tokens = tokenize(line)
#     tokens = map(w -> if w in keys(c.vocab.vocab); w; else; UNK_TOKEN; end, tokens)
#     tokens = filter(w -> rand() < (1 - sqrt(c.vocab.totalWords * c.subsampling_parameter / c.vocab.counts[w])), tokens)
#     tokens
# end

# generate_positive(c::SGCorpus, ch, tokens, pos) = begin
#     win_size = c.params.win_size
#     for offset in -win_size:win_size
#         if offset == 0; continue; end
#         if ((pos + offset) < 1 || (pos + offset) > length(tokens)); continue; end
#         put!(ch, ((tokens[pos], tokens[pos+offset]), 1.))
#         # println("$(tokens[pos])\t$(tokens[pos+offset])\t1")
#     end
# end

# generate_negative(c::SGCorpus, ch, neg_sampler, tokens, pos) = begin
#     for neg in neg_sampler(c.params.neg_samples_per_context)
#         put!(ch, ((tokens[pos], neg), 0.))
#         # println("$(tokens[pos])\t$(neg)\t0")
#     end
# end

in_voc(c::SGCorpus, w) = w in keys(c.vocab.vocab);

discard_token(c::SGCorpus, w) = rand() > (1 - sqrt(c.vocab.totalWords * c.params.subsampling_parameter / c.vocab.counts[w]))

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
        if in_voc(c, tokens[i]) #&& discard_token(c, tokens[i])
            buffer[buffer_pos] = tokens[i]
            buffer_pos += 1
        end
    end
    return buffer[1:buffer_pos-1]
end

get_w_id(c::SGCorpus, w) = c.vocab.vocab[w]

# sigm(x) = (1 ./ (1 + exp.(-x)))
sigm(x::Float32)::Float32 = begin
    if x > 10.
        0.9999546021312976
    elseif x < -10.
        4.5397868702434395e-5
    else
        1 ./ (1 + exp.(-x))
    end
end

update_grads!(  in_grad::SharedArray{Float32,2},
                out_grad::SharedArray{Float32,2},
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


    # the goal of this is not actually to tell that the gradient is unstable
    # this simplifies computation because grad for large values like this is
    # almost 0, so we skip computing almost zero values
    # if abs(act) > 5.
    #     return
    # end
    if abs(act) > 10.
        c = 10. / act
    else
        c = 1.
    end

    act = sigm(act .* label)



    # if isnan(act) || isinf(act)
    #     throw("Activation became infinite")
    # end

    w::Float32 = -label .* (1 .- act) .* lr

    grad_norm::Float32 = 0.

    i = 1
    while i <= n_dims
        in_old = in[i, in_id]
        out_old = out[i, out_id]
        in_grad[i, in_id] -= out_old .* w .* c
        out_grad[i, out_id] -= in_old .* w .* c
        # in[i, in_id] = in_old .- out_old .* w
        # out[i, out_id] = out_old .- in_old .* w
        # grad_norm += out_old .* w .* out_old .* w
        i += 1
    end
    # println(grad_norm)
end



process_context(sample_neg,
                get_buckets,
                w2id,
                shared_params,
                shared_grads,
                win_size::Int32,
                learning_rate::Float32,
                n_dims::Int32,
                tokens,
                pos::Int64
            ) = begin
    context = tokens[pos]
    in_id = w2id(context)
    buckets = get_buckets(context)
    bucket_ind = 1
    n_buckets = length(buckets)
    bucket_lr = @. learning_rate / n_buckets

    # TODO
    # make parallel
    # maybe need to make parallel at a level of process_context
    # to reduce collisions. but nteed to move a lot of stuff into
    # workers. neet to restructure all structures...

    POS_LBL::Float32 = 1.
    NEG_LBL::Float32 = -1.

    # check how much this slows us down
    # win_size = abs(rand(Int)) % win_size + 1

    win_pos = max(1, pos-win_size)
    win_pos_end = min(length(tokens), pos+win_size)
    out_id = -1

    while win_pos <= win_pos_end
        if win_pos == pos; win_pos += 1; continue; end

        out_id = w2id(tokens[win_pos])

        update_grads!(shared_grads.in, shared_grads.out, shared_params.in, shared_params.out, in_id, out_id, POS_LBL, learning_rate, n_dims)

        while bucket_ind <= n_buckets
            update_grads!(shared_grads.buckets, shared_grads.out, shared_params.buckets, shared_params.out, buckets[bucket_ind], out_id, POS_LBL, bucket_lr, n_dims)
            bucket_ind += 1
        end
        win_pos += 1
    end

    # TODO
    # make negative sampling faster
    neg = sample_neg()
    n_neg = length(neg)
    neg_lr = learning_rate # / n_neg
    bucket_neg_lr = bucket_lr # / n_neg
    neg_ind = 1
    neg_out_id = -1
    bucket_ind = 1

    while neg_ind <= n_neg
        neg_out_id = neg[neg_ind]
        update_grads!(shared_grads.in, shared_grads.out, shared_params.in, shared_params.out, in_id, neg_out_id, NEG_LBL, neg_lr, n_dims)

        while bucket_ind <= n_buckets
            update_grads!(shared_grads.buckets, shared_grads.out, shared_params.buckets, shared_params.out, buckets[bucket_ind], neg_out_id, NEG_LBL, bucket_neg_lr, n_dims)
            bucket_ind += 1
        end
        neg_ind += 1
    end

end

process_tokens(c_proc, tokens, learning_rate) = begin
    lr::Float32 = learning_rate / c.params.batch_size
    for pos in 1:length(tokens)
        process_context(tokens, pos, lr)
    end
    length(tokens)
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
            # if abs(act) > 5.
            #     continue
            # end
            loss += -log(sigm(act) + 1e-23) / length(out_ids)
        end

        neg = sample_neg()

        for n in neg
            out_vec = shared_params.out[:, n]
            act = - in_vec' * out_vec
            # if abs(act) > 5.
            #     continue
            # end
            loss += -log(sigm(act) + 1e-23) / length(neg)
        end
    end
    loss
end

get_neg_sampler(c) = begin
    neg_sampler = init_negative_sampling(c.vocab)
    samples_per_context = c.params.neg_samples_per_context
    sample_neg = () -> neg_sampler(samples_per_context)
    sample_neg
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


(c::SGCorpus)(;total_lines=0) = begin

    sample_neg = get_neg_sampler(c)
    get_buckets = get_bucket_fnct(c)
    w2id = get_vocab_fnct(c)
    scheduler = get_scheduler(c)
    eval = (tokens) -> evaluate(c, sample_neg, get_buckets, w2id, tokens)
    c_proc = get_context_processor(c, sample_neg, get_buckets, w2id)

    for epoch in 1:EPOCHS

        seekstart(c.file)

        learning_rate = scheduler(1)

        total_processed = 0

        start = time_ns()
        @time for (ind, line) in enumerate(eachline(c.file))
            tokens = drop_tokens(c, tokenize(line))
            if length(tokens) > 1
                total_processed += process_tokens(c_proc, tokens, learning_rate)
            end

            if total_processed > c.params.batch_size
                apply_grads(c)
                # check_weights(c)
                # println("Updated")
                total_processed = 0
            end

            # if ind % 500 == 0

            # end

            if ind % 100 == 0
                passed_seconds, time_left = compute_lapse(start, ind)
                # print("\rProcessed $ind/$total_lines lines, ")
                learning_rate = scheduler(ind)
                if length(tokens) > 1
                    @printf "\rProcessed %d/%d lines, %dm%ds/%dm%ds, loss %.4f lr %.5f\n" ind total_lines passed_seconds÷60 passed_seconds%60 time_left÷60 time_left%60 eval(tokens)/length(tokens) learning_rate
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
c = SGCorpus(corpus_file, v, learning_rate=0.5, n_buckets=10000)

println("Training Parameters:")
@show c.params

ft = c(total_lines=total_lines)

save_ft(ft, "en_300")
FT.export_for_tb(ft, "en_300")
