using Flux

struct Embedding
    emb_mat
end

Embedding(n_rows::Integer, n_dims::Integer) =
    Embedding(rand(n_rows, n_dims))

(m::Embedding)(idx) = m.emb_mat[idx,:]

Base.getindex(m::Embedding, ind) = m.emb_mat[ind, :]

Flux.@functor Embedding