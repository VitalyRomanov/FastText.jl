using Test
using Revise
includet("SG.jl")

"""
test _compute_in!
"""
@test begin
    in_ = [1. 2. 3. 4.; 4. 3. 2. 1.]'

    b_ = [1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16]'

    in_id = 1
    b_ids = [1 2 3 4]
    n_dims = 4

    buffer = zeros(4)
    _compute_in!(buffer, in_, b_, in_id, b_ids, n_dims)

    isapprox(buffer, (in_[:, 1] + sum(b_, dims=2)[:]) / 5)
end

"""
test _activation
"""
@test begin
    buffer = [1. 2. 3. 4.] / 10.

    out_ = [1. 2. 3. 4.; 5. 6. 7. 8.; 9. 10. 11. 12.; 13. 14. 15. 16.]' / 10.

    out_id = 1
    n_dims = 4
    isapprox(_activation(buffer, out_, out_id, n_dims), sigm(sum(buffer' .* out_[:, 1])))
end

"""
test _update_grads! and _apply_g!
"""
@test begin
    in_g_f =  SharedArray{Bool}(4); in_g_f .= false
    out_g_f =  SharedArray{Bool}(4); out_g_f .= false
    b_g_f =  SharedArray{Bool}(4); b_g_f .=false

    in_g =  SharedArray{Float32}(4, 4); in_g .= 0.
    out_g =  SharedArray{Float32}(4, 4); out_g .= 0.
    b_g =  SharedArray{Float32}(4, 4); b_g .= 0.

    in_ =  SharedArray{Float32}(4, 4); in_ .= 0.
    out_ =  SharedArray{Float32}(4, 4); out_ .= 0.1
    b_ =  SharedArray{Float32}(4, 4); b_ .= 0.

    buffer = [1. 2. 3. 4.] / 10.
    in_[:, 1] = buffer

    out_id = 1; in_id = 1;
    lbl::Float32 = 1.
    lr::Float32 = 1.
    lr_f::Float32 = 1.
    n_dims = 4

    act = _activation(buffer, out_, out_id, n_dims)

    _update_grads!(in_g_f, out_g_f,
                    in_g, out_g,
                    in_, out_,
                    in_id, out_id, lbl, lr, lr_f,
                    n_dims, act)

    altern_in_g = -out_[:,out_id] .* -lbl .* (1 .- act) .* lr * lr_f
    altern_out_g = -in_[:,in_id] .* -lbl .* (1 .- act) .* lr

    grad_calc = in_g_f[1] == true || out_g_f[out_id] == true ||
    isapprox(in_g[:,in_id], altern_in_g) ||
    isapprox(out_g[:,out_id], altern_out_g)

    _apply_g!(in_g_f, in_,
                in_g, n_dims)

    _apply_g!(out_g_f, out_,
            out_g, n_dims)

    isapprox(in_[:, in_id], altern_in_g) ||
    isapprox(out_[:, out_id], altern_out_g) ||
    grad_calc

end

"""
test the correctnes of gradient calculations
"""
test_grads() = begin
    in_g_f =  SharedArray{Bool}(4); in_g_f .= false
    out_g_f =  SharedArray{Bool}(4); out_g_f .= false
    b_g_f =  SharedArray{Bool}(4); b_g_f .= false

    in_g =  SharedArray{Float32}(4, 4); in_g .= 0.
    out_g =  SharedArray{Float32}(4, 4); out_g .= 0.
    b_g =  SharedArray{Float32}(4, 4); b_g .= 0.

    in_ =  SharedArray{Float32}(4, 4); in_[:] = randn(16)
    out_ =  SharedArray{Float32}(4, 4); out_[:] = randn(16)
    b_ =  SharedArray{Float32}(4, 4); b_[:] = randn(16)

    out_id = 3; in_id = 1; neg_id = 2; b_ids = [1,2,3,4]; n_dims = 4
    lbl::Float32 = 1.; lr::Float32 = 1.; lr_f::Float32 = 1/5.
    eps = 0.01

    buffer = zeros(Float32, n_dims)
    wPieces = Dict(in_id => b_ids)
    win_size = 2; n_neg = 1; tokens = [1, 3]; n_tok = 2; pos = 1

    compute_in! = (buffer, in_id, bucket_ids) -> _compute_in!(buffer,
            in_, b_, in_id, bucket_ids, n_dims)
    activation = (buffer, out_id) -> _activation(buffer, out_, out_id, n_dims)
    in_grad_u! = (in_id, out_id, label, lr, lr_factor, act) ->
        _update_grads!(in_g_f, out_g_f, in_g, out_g, in_, out_,
                        in_id, out_id, label, lr, lr_factor, n_dims, act)
    b_grad_u! = (b_id, out_id, label, lr, lr_factor, act) ->
        _update_grads!(b_g_f, out_g_f, b_g, out_g, b_, out_,
                        b_id, out_id, label, lr, lr_factor, n_dims, act)
    neg_sampler = () -> neg_id

    funct = sg_tools(compute_in!, activation, in_grad_u!, b_grad_u!, nothing,
                nothing, neg_sampler, nothing, nothing)

    _process_context(buffer, funct, wPieces, win_size, lr, n_neg, tokens, n_tok, pos)

    loss() = begin
        in_v = (in_[:, in_id] + sum(b_[:, b_ids], dims=2)[:]) / (1 + length(b_ids))
        out_v = out_[:, out_id]
        neg_v = out_[:, neg_id]
        -log(sigm(sum(in_v .* out_v))) - log(sigm(-sum(in_v .* neg_v)))
    end

    in_g_der = copy(in_g); b_g_der = copy(b_g); out_g_der = copy(out_g);
    in_g .= 0.; b_g .= 0.; out_g .= 0.;

    for (p, p_grad) in [(in_, in_g), (b_, b_g), (out_, out_g)]
        for i in 1:length(p)
            init_p = p[i]
            p[i] += eps; loss2 = loss()
            p[i] -= 2*eps; loss1 = loss()
            p_grad[i] = (loss2 - loss1) / (2*eps)
            p[i] = init_p
        end
    end

    calc_err(grd, est_grd) = begin
        err = grd .- est_grd
        div = copy(grd)
        div = collect(map(x -> if x==0.; 1. else x end, div))
        err ./= div
        sum(err) / length(err)
    end

    err = (calc_err(in_g_der, in_g) + calc_err(out_g_der, out_g) + calc_err(b_g_der, b_g))/3
    @show err
    @show in_g
    @show in_g_der
    @show b_g
    @show b_g_der
    @show out_g
    @show out_g_der
    return err
end
@test test_grads() < 0.01

"""
test vocabulary
"""
test_vocabulary() = begin
    v = Vocab()
    tokens = ["b","b","b","b","c","c","c","d","d","e", "a","a","a","a","a"]
    learnVocab!(v, tokens)
    @test v.vocab["a"] == 5 || v.vocab["b"] == 1 || v.vocab["c"] == 2 ||
          v.vocab["d"] == 3 || v.vocab["e"] == 4
    @test v.counts["a"] == 5 || v.counts["b"] == 4 || v.counts["c"] == 3 ||
        v.counts["d"] == 2 || v.counts["e"] == 1

    v = prune(v,3)
    @test v.vocab["a"] == 1 || v.vocab["b"] == 2 || v.vocab["c"] == 3
    @test v.counts["a"] == 5 || v.counts["b"] == 4 || v.counts["c"] == 3
    @test length(v) == 3
    @test v.totalWords == 15

end
test_vocabulary()

"""
test negative sampling
"""
test_ns() = begin
    v = Vocab()
    # tokens = ["b","b","b","b","c","c","c","d","d","e", "a","a","a","a","a"]
    tokens = tokenize(read("wiki_00", String))
    learnVocab!(v, tokens)
    prune(v, length(v), 1)

    # smpl_neg = init_negative_sampling(v)
    smpl_neg = init_negative_sampling_bisect(v)
    # @show sizeof(smpl_neg)

    n_experiments = v.totalWords * 20
    bins = zeros(Float32, length(v))
    for i = 1:n_experiments
        bins[smpl_neg()] += 1
    end
    bins ./= n_experiments

    ordered_words = sort(collect(v.vocab), by=x->x[2])
    probs = zeros(length(ordered_words))
    for (w, id) in ordered_words
        probs[id] = v.counts[w] / v.totalWords
    end
    probs .^= 3/4
    probs ./= sum(probs)
    err = sum(abs.(probs - bins) ./ probs) / length(v)
    # err = maximum(abs.(probs - bins))
    # ind = argmax(abs.(probs - bins) ./ probs)
    # @show err
    # @show probs
    # @show bins
    # @show sum(bins)
    ## @show [v.counts["a"], v.counts["a"], v.counts["a"], v.counts["a"], v.counts["a"]]
    # @show v.vocab["a"]
    # @show v.vocab["b"]
    # @show v.vocab["c"]
    # @show v.vocab["d"]
    # @show v.vocab["e"]
    @test err < 0.1
    # @show err
    # @show probs[ind], abs(probs[ind] - bins[ind]) / probs[ind]
end
test_ns()

"""
test bisect_left
"""
test_bisect_left() = begin
    arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]

    @test bisect_left(arr, 1, 1, length(arr)) == 1
    @test bisect_left(arr, 2, 1, length(arr)) == 2
    @test bisect_left(arr, 3, 1, length(arr)) == 3
    @test bisect_left(arr, 4, 1, length(arr)) == 4
    @test bisect_left(arr, 5, 1, length(arr)) == 5
    @test bisect_left(arr, 6, 1, length(arr)) == 6
    @test bisect_left(arr, 7, 1, length(arr)) == 7
    @test bisect_left(arr, 8, 1, length(arr)) == 8
    @test bisect_left(arr, 9, 1, length(arr)) == 9
    @test bisect_left(arr, 10, 1, length(arr)) == 10
    @test bisect_left(arr, 11, 1, length(arr)) == 11
    @test bisect_left(arr, 12, 1, length(arr)) == 11
    @test bisect_left(arr, 13, 1, length(arr)) == 11
    @test bisect_left(arr, 15, 1, length(arr)) == 11
    @test bisect_left(arr, 15, 1, length(arr)) == 11
end
test_bisect_left()


"""
test Facebook compatible hash
"""
@test fb_hash("a") == 3826002220
@test fb_hash("ф") == 1170258024
@test fb_hash("<a") == 1761008750
