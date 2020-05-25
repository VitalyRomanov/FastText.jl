using Test
using Revise
include("SG.jl")

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
    b_g =  SharedArray{Float32}(4, 4); b_g_f .= 0.

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
