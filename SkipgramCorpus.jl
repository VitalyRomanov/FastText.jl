module SkipGramCorpus

struct SGCorpus
    file
    vocab
    win_size
    contexts_in_batch
    neg_samples_per_context
    subsampling_parameter
end

SGCorpus(file, vocab; win_size=5, contexts_in_batch=15, neg_samples_per_context=15, subsampling_parameter=1e-4)