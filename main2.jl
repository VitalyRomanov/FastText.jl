# using Pkg
# Pkg.activate(".")
# Pkg.add("Revise")
# Pkg.add("StatsBase")
# Pkg.add("JLD2")
using Revise


includet("SG.jl")

# using Profile

# FILENAME = "wiki_00"
EPOCHS = 1
# FILENAME = "test.txt"
# FILENAME = "/Users/LTV/Desktop/AA_t.txt"
FILENAME = "/home/ltv/data/local_run/wikipedia/extracted/en_wiki_plain/AA.txt"
# FILENAME = "/Volumes/External/datasets/Language/Corpus/en/en_wiki_tiny/wiki_tiny.txt"



# TODO
# implement PMI based word Ngram extraction
# https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf


# process_context(buffer, sample_neg, get_buckets, w2id, shared_params, shared_grads,
#                 win_size, lr, n_dims, tokens, pos) = begin
#     # current context
#     context = tokens[pos]
#     in_id = w2id(context)
#     buckets = get_buckets(context)
#
#     # prepare to iterate buckets
#     bucket_ind = 1
#     n_buckets = length(buckets)
#
#     # consts
#     POS_LBL::Float32 = 1; NEG_LBL::Float32 = -1.
#
#     # define window region
#     win_pos = max(1, pos-win_size); win_pos_end = min(length(tokens), pos+win_size)
#
#     # get pointers, it is easier to pass them to other functions
#     in_ = shared_params.in; out_ = shared_params.out; b_ = shared_params.buckets
#     in_g = shared_grads.in; out_g = shared_grads.out; b_g = shared_grads.buckets
#
#     compute_in!(buffer, in_, b_, in_id, buckets, n_dims)
#
#     # init
#     loss = 0.
#     act = 0.
#     processed = 0
#
#     lr_factor::Float32 = 1. / (1 + length(buckets))
#
#     while win_pos <= win_pos_end
#         if win_pos == pos; win_pos += 1; continue; end
#
#         out_target = tokens[win_pos]
#         # println("$context\t$out_target\t$POS_LBL")
#         out_id = w2id(out_target)
#
#         # act = ft_act(in_, b_, out_, in_id, buckets, out_id, n_dims)
#         act = activation(buffer, out_, out_id, n_dims)
#         loss += -log(act)
#         processed += 1
#
#         update_grads!(in_g, out_g, in_, out_,
#                                     in_id, out_id, POS_LBL, lr, lr_factor, n_dims, act)
#
#         while bucket_ind <= n_buckets
#             update_grads!(b_g, out_g, b_, out_,
#                             buckets[bucket_ind], out_id, POS_LBL, lr, lr_factor, n_dims, act)
#             bucket_ind += 1
#         end
#         win_pos += 1
#     end
#
#     # TODO
#     # make negative sampling faster
#     # neg = sample_neg()
#     n_neg = 15#length(neg)
#     neg_ind = 1
#     bucket_ind = 1
#
#     while neg_ind <= n_neg
#         # neg_out = sample_neg() #neg[neg_ind]
#         # println("$context\t$neg_out\t$NEG_LBL")
#         neg_out_id = sample_neg() #w2id(neg_out)
#
#         # act = ft_act(in_, b_, out_, in_id, buckets, neg_out_id, n_dims)
#         act = activation(buffer, out_, neg_out_id, n_dims)
#         loss += -log(1-act)
#         processed += 1
#
#         update_grads!(in_g, out_g, in_, out_,
#                             in_id, neg_out_id, NEG_LBL, lr, lr_factor, n_dims, 1-act)
#
#         while bucket_ind <= n_buckets
#             update_grads!(b_g, out_g, b_, out_,
#                         buckets[bucket_ind], neg_out_id, NEG_LBL, lr, lr_factor, n_dims, 1-act)
#             bucket_ind += 1
#         end
#         neg_ind += 1
#     end
#     loss, processed
# end


process_tokens(c_proc, tokens, n_tokens, learning_rate) = begin
    lr::Float32 = learning_rate #/ c.params.batch_size
    loss = 0.
    processed = 0
    for pos in 1:n_tokens
        l, p = c_proc(tokens, n_tokens, pos, lr)
        loss += l
        processed += p
    end
    loss, processed
end

learn_voc(file, voc_size) = begin

    v = Vocab()

    total_lines = 0
    print("Learning vocabulary...")
    for (ind, line) in enumerate(eachline(corpus_file))
        # global total_lines
        # if length(v) < voc_size * 3
            tokens = tokenize(line)
            learnVocab!(v, tokens)
        # end
        total_lines = ind
    end
    println("done")

    v = prune(v, voc_size)
    v, total_lines
end

corpus_file = open(FILENAME)

v, total_lines = learn_voc(corpus_file, 50000)

println("Begin training")
c = SGCorpus(corpus_file, v, learning_rate=1e-3, n_buckets=5000, neg_samples_per_context=20)

println("Training Parameters:")
@show c.params

# using Juno
ft = c(total_lines=total_lines)

save_ft(ft, "en_300")
FT.export_for_tb(ft, "en_300")
FT.export_w2v(ft, "emb.txt")
