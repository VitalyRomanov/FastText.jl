using Revise
includet("SG.jl")

test_activation() = begin
    n_dims = 300
    v_size = 10000
    n_iter = 1000000

    buffer = rand(n_dims)
    out_ = rand(n_dims, v_size)

    out_id = 1
    n_dims = 4


    i = 1
    while i <= n_iter
        out_id = abs(rand(Int64)) % v_size + 1
        _activation(buffer, out_, out_id, n_dims)
        i+= 1
    end
end
@time test_activation()


test_compute_in() = begin
    n_dims = 300
    v_size = 10000
    n_buckets = 1000
    n_iter = 100000

    buffer = rand(n_dims)
    in_ = rand(n_dims, v_size)
    b_ = rand(n_dims, n_buckets)

    in_id = 1
    b_ids = [1; 2; 3; 4]

    i = 1
    while i <= n_iter
        _compute_in!(buffer, in_, b_, in_id, b_ids, n_dims)
        i+= 1
    end
end
@time test_compute_in()

test_negative_sampling() = begin
    v = Vocab()
    tokens = tokenize(read("wiki_00", String))
    # tokens = ["b","b","b","b","c","c","c","d","d","e", "a","a","a","a","a"]
    learnVocab!(v, tokens)
    prune(v, 5)

    do_test() = begin

        smpl_neg = init_negative_sampling(v)

        n_iter = 1000000

        i = 1
        while i <= n_iter
            smpl_neg()
            i += 1
        end
    end
    @time do_test()
end
test_negative_sampling()

test_apply_g() = begin
    n_dims = 300
    v_size = 10000
    n_buckets = 1000
    n_iter = 1000

    atomic = ones(Bool, v_size)
    atomic .= true

    par = rand(n_dims, v_size)
    grad = rand(n_dims, v_size)

    i = 1
    while i <= n_iter
        _apply_g!(atomic, par, grad, n_dims)
        i += 1
        atomic .= true
    end
end
@time test_apply_g()
