cd("/Users/LTV/dev/FastText.jl/")
include("FastText.jl")

EMB_SIZE = 15
VOC_SIZE = 10
BUCKET_SIZE = 10

input_index = [5, 5, 5, 5]
output_indices = [1, 2, 3, 5]

data = []
for i in 1:4
    push!(data, ([input_index[i], output_indices[i]], 1.))
end

ft = FastText(VOC_SIZE, EMB_SIZE, BUCKET_SIZE)

loss(x,y) = Flux.logitbinarycrossentropy(ft.in[x[1]]' * ft.out[x[2]], y)

opt = Descent(0.3)

for _ in 1:100
    Flux.train!(loss, params(ft), data, opt)
end







