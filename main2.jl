# module SkipGramCorpus
# cd("/Users/LTV/dev/FastText.jl/")
# include("Vocab.jl")
using Revise
includet("LanguageTools.jl")
includet("FastText.jl")
includet("SG.jl")

using .FT
using SharedArrays

# FILENAME = "wiki_01"
EPOCHS = 1
# FILENAME = "test.txt"
FILENAME = "/Users/LTV/Desktop/AA_t.txt"
FILENAME = "/home/ltv/data/local_run/wikipedia/extracted/en_wiki_plain/AA.txt"
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

discard_token(c::SGCorpus, w) = rand() < (1 - sqrt(c.vocab.totalWords * c.params.subsampling_parameter / c.vocab.counts[w]))

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
        if in_voc(c, tokens[i]) && !discard_token(c, tokens[i])
            buffer[buffer_pos] = tokens[i]
            buffer_pos += 1
        end
    end
    return buffer[1:buffer_pos-1]
end

sigm(x) = (1 ./ (1 + exp.(-x)))

update_grads!(in_grad::SharedArray{Float32,2}, out_grad::SharedArray{Float32,2},
                in::SharedArray{Float32,2}, out::SharedArray{Float32,2},
                in_id::Int64, out_id::Int64, label::Float32, lr::Float32, lr_factor::Float32,
                n_dims::Int32, act::Float32) = begin

    w::Float32 = -label .* (1 .- act) .* lr

    i = 1
    while i <= n_dims
        in_old = in[i, in_id]
        out_old = out[i, out_id]
        in_grad[i, in_id] -= out_old .* w .* lr_factor
        out_grad[i, out_id] -= in_old .* w
        i += 1
    end
end

ft_act(in, b, out, in_id, b_id, out_id, n_dims)::Float32 = begin
    a = 0.
    n_bs = length(b_id)
    factor = 1. ./ (1. + n_bs)
    i = 1
    while i <= n_dims
        temp = in[i, in_id]
        b_i = 1
        while b_i <= n_bs
            temp += b[i, b_id[b_i]]
            b_i += 1
        end
        a += temp .* factor .* out[i, out_id]
        i += 1
    end
    sigm(a)
end

process_context(sample_neg, get_buckets, w2id, shared_params, shared_grads,
                win_size, lr, n_dims, tokens, pos) = begin
    # current context
    context = tokens[pos]
    in_id = w2id(context)
    buckets = get_buckets(context)

    # prepare to iterate buckets
    bucket_ind = 1
    n_buckets = length(buckets)

    # consts
    POS_LBL::Float32 = 1; NEG_LBL::Float32 = -1.

    # define window region
    win_pos = max(1, pos-win_size); win_pos_end = min(length(tokens), pos+win_size)

    # get pointers
    in_ = shared_params.in; out_ = shared_params.out; b_ = shared_params.buckets
    in_g = shared_grads.in; out_g = shared_grads.out; b_g = shared_grads.buckets

    # init
    loss = 0.
    act = 0.
    processed = 0

    lr_factor::Float32 = 1. / (1 + length(buckets))

    while win_pos <= win_pos_end
        if win_pos == pos; win_pos += 1; continue; end

        out_target = tokens[win_pos]
        # println("$context\t$out_target\t$POS_LBL")
        out_id = w2id(out_target)

        act = ft_act(in_, b_, out_, in_id, buckets, out_id, n_dims)
        loss += -log(act)
        processed += 1

        update_grads!(in_g, out_g, in_, out_,
                                    in_id, out_id, POS_LBL, lr, lr_factor, n_dims, act)

        while bucket_ind <= n_buckets
            update_grads!(b_g, out_g, b_, out_,
                            buckets[bucket_ind], out_id, POS_LBL, lr, lr_factor, n_dims, act)
            bucket_ind += 1
        end
        win_pos += 1
    end

    # TODO
    # make negative sampling faster
    neg = sample_neg()
    n_neg = length(neg)
    neg_ind = 1
    bucket_ind = 1

    while neg_ind <= n_neg
        neg_out = neg[neg_ind]
        # println("$context\t$neg_out\t$NEG_LBL")
        neg_out_id = w2id(neg_out)

        act = ft_act(in_, b_, out_, in_id, buckets, neg_out_id, n_dims)
        loss += -log(1-act)
        processed += 1

        update_grads!(in_g, out_g, in_, out_,
                            in_id, neg_out_id, NEG_LBL, lr, lr_factor, n_dims, 1-act)

        while bucket_ind <= n_buckets
            update_grads!(b_g, out_g, b_, out_,
                        buckets[bucket_ind], neg_out_id, NEG_LBL, lr, lr_factor, n_dims, 1-act)
            bucket_ind += 1
        end
        neg_ind += 1
    end
    loss, processed
end

process_tokens(c_proc, tokens, learning_rate) = begin
    lr::Float32 = learning_rate / c.params.batch_size
    loss = 0.
    processed = 0
    for pos in 1:length(tokens)
        l, p = c_proc(tokens, pos, lr)
        loss += l
        processed += p
    end
    loss, processed
end


(c::SGCorpus)(;total_lines=0) = begin

    sample_neg = get_neg_sampler(c)
    get_buckets = get_bucket_fnct(c)
    w2id = get_vocab_fnct(c)
    scheduler = get_scheduler(c)
    eval = (tokens) -> 1. #evaluate(c, sample_neg, get_buckets, w2id, tokens)
    c_proc = get_context_processor(c, sample_neg, get_buckets, w2id)

    for epoch in 1:EPOCHS

        seekstart(c.file)

        learning_rate = c.params.learning_rate

        total_processed = 0
        loss = 0.

        start = time_ns()
        @time for (ind, line) in enumerate(eachline(c.file))
            tokens = drop_tokens(c, tokenize(line))
            # @show tokens
            if length(tokens) > 1
                l, t = process_tokens(c_proc, tokens, learning_rate)
                loss += l; total_processed += t
            end

            if ind % 100 == 0
                passed_seconds, time_left = compute_lapse(start, ind)
                if length(tokens) > 1
                    @printf "\rProcessed %d/%d lines, %dm%ds/%dm%ds, loss %.4f lr %.5f\n" ind total_lines passed_seconds÷60 passed_seconds%60 time_left÷60 time_left%60 loss/total_processed learning_rate
                end
            end

            if total_processed > c.params.batch_size
                apply_grads(c)
                check_weights(c)
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

    FastText(
        in_m,
        out_m,
        bucket_m,
        c.vocab,
        c.params.min_ngram,
        c.params.max_ngram,
    )
end

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

v = prune(v, 10000)

println("Begin training")
c = SGCorpus(corpus_file, v, learning_rate=0.01, n_buckets=20000)

println("Training Parameters:")
@show c.params

ft = c(total_lines=total_lines)

save_ft(ft, "en_300")
FT.export_for_tb(ft, "en_300")
