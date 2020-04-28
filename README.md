Lessons

- Prefer column wise indexing
- `for i = 1:100` is slower than while loop because it allocates memory
- any slice, even @views will allocate memory
- @inbounds does not help much in performance
- apparently accessing fields of mutable (or even immutable structs) is slow
- gensim uses lookup tables for complex functions https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/word2vec_inner.pyx
- exp table size EXP_TABLE_SIZE = 1000, max exp value MAX_EXP = 6
- gensim also uses random window reduction window, which makes it faster - simply sample window size
- gensim skips samples where activation is too high https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/word2vec_inner.pyx line 122
- algorithm for reading binary format
- looking at fasttext, it seems they used quantizations, that is why htey are fast https://github.com/facebookresearch/fastText/blob/master/src/quantmatrix.cc
- gensim skips central word in window https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/word2vec_corpusfile.pyx line 324
- but then they add it in neg sampling as positive. if word is its own negative - sample is skipped https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/word2vec_inner.pyx line 217
- it seems there is no gradient for positive words https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/word2vec_inner.pyx line 245
- gensim takes a context word (center) and words in the window. For each word in window they generate negative samples and one positive. 
- their gradient formula seems to be a little bit different (label - f) * alpha where f is sigmoid and labels are 0 and 1
- do not update central word params immediately!!! gradients will oscillate
- gradients clipped only for negative samles, not for positive 