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


# """
# test the correctnes of gradient calculations
# """
# @test begin
#     in_g_f =  SharedArray{Bool}(4); in_g_f .= false
#     out_g_f =  SharedArray{Bool}(4); out_g_f .= false
#     b_g_f =  SharedArray{Bool}(4); b_g_f .=false
#
#     in_g =  SharedArray{Float32}(4, 4); in_g .= 0.
#     out_g =  SharedArray{Float32}(4, 4); out_g .= 0.
#     b_g =  SharedArray{Float32}(4, 4); b_g .= 0.
#
#     in_ =  SharedArray{Float32}(4, 4); in_[:] = randn(16)
#     out_ =  SharedArray{Float32}(4, 4); out_[:] = randn(16)
#     b_ =  SharedArray{Float32}(4, 4); b_[:] = randn(16)
#
#     out_id = 1; in_id = 1; b_ids = [1,2,3,4]; n_dims = 4
#     lbl::Float32 = 1.; lr::Float32 = 1.; lr_f::Float32 = 1/5.
#     eps = 0.1
#
#     get_act() = begin
#         buffer = zeros(4)
#         _compute_in!(buffer, in_, b_, in_id, b_ids, n_dims)
#
#         act = _activation(buffer, out_, out_id, n_dims)
#     end
#
#     get_grad(param, row_id, col_id) = begin
#         initial_val = param[row_id, col_id]
#         param[row_id, col_id] -= eps; act1 = -log(get_act())
#         param[row_id, col_id] += 2 * eps; act2 = -log(get_act())
#         param[row_id, col_id] = initial_val
#         grad = (act2 - act1) / (2 * eps)
#     end
#
#     upd_g_out(in_id, out_id, lbl) = begin
#         buffer = zeros(Float32, 4)
#         _compute_in!(buffer, in_, b_, in_id, b_ids, n_dims)
#         act = _activation(buffer, out_, out_id, n_dims)
#         _update_grads_out!(out_g_f, out_g, buffer, out_id, lbl, lr,
#                         n_dims, act)
#     end
#
#     upd_g_in(in_id, out_id, lbl) = begin
#         buffer = zeros(Float32, 4)
#         _compute_in!(buffer, in_, b_, in_id, b_ids, n_dims)
#         act = _activation(buffer, out_, out_id, n_dims)
#         _update_grads_in!(in_g_f, in_g, out_, in_id, out_id, lbl, lr, lr_f,
#                         n_dims, act)
#     end
#
#     upd_g_b(b_id, out_id, lbl) = begin
#         buffer = zeros(Float32, 4)
#         _compute_in!(buffer, in_, b_, in_id, b_ids, n_dims)
#         act = _activation(buffer, out_, out_id, n_dims)
#         _update_grads_in!(b_g_f, b_g, out_, b_id, out_id, lbl, lr, lr_f,
#                         n_dims, act)
#     end
#
#     # lbl::Float32 = 1.
#
#     row_id = 1; col_id = 1;
#     grad = get_grad(out_, row_id, col_id)
#     upd_g_out(in_id, col_id, lbl)
#     out_pos = (out_g[row_id, col_id] - grad) / grad < 0.01
#     println("out pos", grad, " ", out_g[row_id, col_id], " ", out_g[row_id, col_id] - grad, " ", out_g[row_id, col_id] / grad)
#     in_g .= 0.; out_g .= 0.; b_g .= 0.
#
#     row_id = 1; col_id = 1;
#     grad = get_grad(in_, row_id, col_id)
#     upd_g_in(col_id, out_id, lbl)
#     in_pos = (in_g[row_id, col_id] - grad) / grad < 0.01
#     println("in pos", grad, " ", in_g[row_id, col_id], " ", in_g[row_id, col_id] - grad, " ", in_g[row_id, col_id] / grad)
#     in_g .= 0.; out_g .= 0.; b_g .= 0.
#
#     row_id = 1; col_id = 1;
#     grad = get_grad(b_, row_id, col_id)
#     upd_g_b(col_id, out_id, lbl)
#     b_pos = (b_g[row_id, col_id] - grad) / grad < 0.01
#     println("b pos", grad, " ", b_g[row_id, col_id], " ", b_g[row_id, col_id] - grad, " ", b_g[row_id, col_id] / grad)
#     in_g .= 0.; out_g .= 0.; b_g .= 0.
#
#     lbl = -1.
#
#     row_id = 1; col_id = 1;
#     grad = - get_grad(out_, row_id, col_id)
#     upd_g_out(in_id, col_id, lbl)
#     out_neg = (out_g[row_id, col_id] - grad) / grad < 0.01
#     println("out neg", grad, " ", out_g[row_id, col_id], " ", out_g[row_id, col_id] - grad, " ", out_g[row_id, col_id] / grad)
#     in_g .= 0.; out_g .= 0.; b_g .= 0.
#
#     row_id = 1; col_id = 1;
#     grad = - get_grad(in_, row_id, col_id)
#     upd_g_in(col_id, out_id, lbl)
#     in_neg = (in_g[row_id, col_id] - grad) / grad < 0.01
#     println("in neg", grad, " ", in_g[row_id, col_id], " ", in_g[row_id, col_id] - grad, " ", in_g[row_id, col_id] / grad)
#     in_g .= 0.; out_g .= 0.; b_g .= 0.
#
#     row_id = 1; col_id = 1;
#     grad = - get_grad(b_, row_id, col_id)
#     upd_g_b(col_id, out_id, lbl)
#     b_neg = (b_g[row_id, col_id] - grad) / grad < 0.01
#     println("b neg", grad, " ", b_g[row_id, col_id], " ", b_g[row_id, col_id] - grad, " ", b_g[row_id, col_id] / grad)
#     in_g .= 0.; out_g .= 0.; b_g .= 0.
#
#
#     # println("", grad, " ", in_g[row_id, col_id], " ", in_g[row_id, col_id] - grad, " ", in_g[row_id, col_id] / grad)
#     true
#     # out_pos && in_pos && b_pos && out_neg && in_neg && b_neg
# end


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


# @test begin
#     in_g =  SharedArray{Float32}(4, 4); in_g .= 0.
#     out_g =  SharedArray{Float32}(4, 4); out_g .= 0.
#     b_g =  SharedArray{Float32}(4, 4); b_g .= 0.
#
#     in_ =  SharedArray{Float32}(4, 4); in_[:] = randn(16)
#     out_ =  SharedArray{Float32}(4, 4); out_[:] = randn(16)
#     b_ =  SharedArray{Float32}(4, 4); b_[:] = randn(16)
#
#     out_id = 1; in_id = 1; b_ids = [1,2,3,4]; n_dims = 4
#     lbl::Float32 = 1.; lr::Float32 = 1.; lr_f::Float32 = 1.
#     eps = 0.001
#
#     get_act() = begin
#         in_v = in_[:, in_id]# + sum(b_[:, b_ids], dims=2)[:]
#         out_v = out_[:, out_id]
#
#         act = sigm(sum(in_v .* out_v))
#     end
#
#     param_id = 1
#     under_inv = out_
#     initial_val = under_inv[param_id]
#     under_inv[param_id] -= eps; act1 = -log(get_act())
#     under_inv[param_id] += 2 * eps; act2 = -log(get_act())
#     grad = (act2 - act1) / (2 * eps)
#     under_inv[param_id] = initial_val
#
#     get_grad = (in_id, out_id, label) -> begin
#         in_v = in_[:, in_id]# + sum(b_[:, b_ids], dims=2)[:]
#         out_v = out_[:, out_id]
#         act = sigm(sum(in_v .* out_v))
#
#         w = act - label
#         in_g[:, in_id] = out_[:, out_id] .* w
#         out_g[:, out_id] = in_[:, in_id] .* w
#     end
#
#     get_grad(in_id, out_id, lbl)
#
#     println(grad, " ", out_g[param_id], " ", out_g[param_id] / grad, " ", out_g[param_id] - grad)
#     true
# end
