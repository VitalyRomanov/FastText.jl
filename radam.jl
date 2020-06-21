"""
f_m: first moment
f_m: second moment
grad: gradients
par: weights to update
i: word id
beta1: momentum for the first moment
beta2: momentum for the second moment
alpha: learning rate
p_infty: parameter?
"""
update_radam!(f_m, s_m, grad, par, i, iter, beta1, beta2, alpha, p_infty, n_dims) = begin
    # https://medium.com/@lessw/new-state-of-the-art-ai-optimizer-rectified-adam-radam-5d854730807b
    d = 1
    while d <= n_dims
        f_m[d, i] = @. beta1 * f_m[d, i] + (1 - beta1) * grad[d, i]
        s_m[d, i] = @. beta2 * s_m[d, i] + (1 - beta2) * grad[d, i] ^ 2
        f_m_corr = 1 / (1 - beta1 ^ iter[i]); c_f_m = f_m[d, i] * f_m_corr
        p = p_infty - 2 * iter[i] * beta2 ^ iter[i] / (1 - beta2 ^ iter[i])
        if p > 4.
            r = sqrt((p - 4) * (p - 2) * p_infty / ((p_infty - 4) * (p_infty - 2) * p))
            s_m_corr = 1 / (1 - beta2 ^ iter[i]); c_s_m = sqrt(s_m[d, i] * s_m_corr)
            par[d, i] -= r * alpha * c_f_m / (c_s_m + 1e-8)
        else
            par[d, i] -= c_f_m * alpha
        end
        grad[d, i] = 0.
        d += 1
    end
    iter[i] += 1
end

_apply_g_radam!(atomic, par, grad, n_dims, f_m, s_m, iter, beta1, beta2, alpha, p_infty) = begin
    n = length(atomic)
    # p_infty = 2 / (1 - beta2) - 1
    i = 1
    while i <= n
        if atomic[i] == false; i += 1; continue; end

        update_radam!(f_m, s_m, grad, par, i, iter, beta1, beta2, alpha, p_infty, n_dims)

        atomic[i] = false
        i += 1
    end
end

apply_grads_radam(c, alpha) = begin
    beta1 = 0.9
    beta2 = 0.999
    p_infty = 2 / (1 - beta2) - 1
    _apply_g_radam!(c.shared_grads.atomic_in, c.shared_params.in,
                c.shared_grads.in, c.params.n_dims, c.shared_moments.in_f_m, c.shared_moments.in_s_m, c.shared_moments.in_iter, beta1, beta2, alpha, p_infty)
    _apply_g_radam!(c.shared_grads.atomic_out, c.shared_params.out,
                c.shared_grads.out, c.params.n_dims, c.shared_moments.out_f_m, c.shared_moments.out_s_m, c.shared_moments.out_iter, beta1, beta2, alpha, p_infty)
    _apply_g_radam!(c.shared_grads.atomic_buckets, c.shared_params.buckets,
                c.shared_grads.buckets, c.params.n_dims, c.shared_moments.bucket_f_m, c.shared_moments.bucket_s_m, c.shared_moments.bucket_iter, beta1, beta2, alpha, p_infty)
end
