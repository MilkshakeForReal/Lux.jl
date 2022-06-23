
"""
    Dense(in_dims => out_dims, activation=identity; init_weight=glorot_uniform,
          init_bias=zeros32, bias::Bool=true)

Create a traditional fully connected layer, whose forward pass is given by:
`y = activation.(weight * x .+ bias)`

## Arguments

  - `in_dims`: number of input dimensions
  - `out_dims`: number of output dimensions
  - `activation`: activation function

## Keyword Arguments

  - `init_weight`: initializer for the weight matrix
    (`weight = init_weight(rng, out_dims, in_dims)`)
  - `init_bias`: initializer for the bias vector (ignored if `bias=false`)
  - `bias`: whether to include a bias vector

## Input

  - `x` must be a Matrix of size `in_dims × B` or a Vector of length `in_dims`

## Returns

  - Matrix of size `out_dims × B` or a Vector of length `out_dims`
  - Empty `NamedTuple()`

## Parameters

  - `weight`: Weight Matrix of size `out_dims × in_dims`
  - `bias`: Bias of size `out_dims × 1` (present if `bias=true`)
"""
struct Dense{bias, F1, F2, F3} <: AbstractExplicitLayer
    activation::F1
    in_dims::Int
    out_dims::Int
    init_weight::F2
    init_bias::F3
end

function Base.show(io::IO, d::Dense{bias}) where {bias}
    print(io, "Dense($(d.in_dims) => $(d.out_dims)")
    (d.activation == identity) || print(io, ", $(d.activation)")
    bias || print(io, ", bias=false")
    return print(io, ")")
end

function Dense(mapping::Pair{<:Int, <:Int}, activation=identity; init_weight=glorot_uniform,
               init_bias=zeros32, bias::Bool=true)
    return Dense(first(mapping), last(mapping), activation; init_weight=init_weight,
                 init_bias=init_bias, bias=bias)
end

function Dense(in_dims::Int, out_dims::Int, activation=identity; init_weight=glorot_uniform,
               init_bias=zeros32, bias::Bool=true)
    activation = NNlib.fast_act(activation)
    return Dense{bias, typeof(activation), typeof(init_weight), typeof(init_bias)}(activation,
                                                                                   in_dims,
                                                                                   out_dims,
                                                                                   init_weight,
                                                                                   init_bias)
end

function initialparameters(rng::AbstractRNG, d::Dense{bias}) where {bias}
    if bias
        return (weight=d.init_weight(rng, d.out_dims, d.in_dims),
                bias=d.init_bias(rng, d.out_dims, 1))
    else
        return (weight=d.init_weight(rng, d.out_dims, d.in_dims),)
    end
end

function parameterlength(d::Dense{bias}) where {bias}
    return bias ? d.out_dims * (d.in_dims + 1) : d.out_dims * d.in_dims
end
statelength(d::Dense) = 0

@inline function (d::Dense{false})(x::AbstractVecOrMat, ps, st::NamedTuple)
    return applyactivation(d.activation, ps.weight * x), st
end

@inline function (d::Dense{false, typeof(identity)})(x::AbstractVecOrMat, ps,
                                                     st::NamedTuple)
    return ps.weight * x, st
end

@inline function (d::Dense{false})(x::AbstractArray, ps, st::NamedTuple)
    sz = size(x)
    x_reshaped = reshape(x, sz[1], :)
    return reshape(applyactivation(d.activation, ps.weight * x_reshaped), d.out_dims,
                   sz[2:end]...), st
end

@inline function (d::Dense{false, typeof(identity)})(x::AbstractArray, ps, st::NamedTuple)
    sz = size(x)
    x_reshaped = reshape(x, sz[1], :)
    return reshape(ps.weight * x_reshaped, d.out_dims, sz[2:end]...), st
end

@inline function (d::Dense{true})(x::AbstractVector, ps, st::NamedTuple)
    return applyactivation(d.activation, elementwise_add(ps.weight * x, vec(ps.bias))), st
end

@inline function (d::Dense{true, typeof(identity)})(x::AbstractVector, ps, st::NamedTuple)
    return elementwise_add(ps.weight * x, vec(ps.bias)), st
end

@inline function (d::Dense{true})(x::AbstractMatrix, ps, st::NamedTuple)
    return applyactivation(d.activation, elementwise_add(ps.weight * x, ps.bias)), st
end

@inline function (d::Dense{true, typeof(identity)})(x::AbstractMatrix, ps, st::NamedTuple)
    return elementwise_add(ps.weight * x, ps.bias), st
end

@inline function (d::Dense{true})(x::AbstractArray, ps, st::NamedTuple)
    sz = size(x)
    x_reshaped = reshape(x, sz[1], :)
    return (reshape(applyactivation(d.activation,
                                    elementwise_add(ps.weight * x_reshaped, ps.bias)),
                    d.out_dims, sz[2:end]...),
            st)
end

@inline function (d::Dense{true, typeof(identity)})(x::AbstractArray, ps, st::NamedTuple)
    sz = size(x)
    x_reshaped = reshape(x, sz[1], :)
    return (reshape(elementwise_add(ps.weight * x_reshaped, ps.bias), d.out_dims,
                    sz[2:end]...), st)
end

"""
    Scale(dims, activation=identity; init_weight=ones32, init_bias=zeros32, bias::Bool=true)

Create a Sparsely Connected Layer with a very specific structure (only Diagonal
Elements are non-zero). The forward pass is given by: `y = activation.(weight .* x .+ bias)`

## Arguments

  - `dims`: size of the learnable scale and bias parameters.
  - `activation`: activation function

## Keyword Arguments

  - `init_weight`: initializer for the weight matrix
    (`weight = init_weight(rng, out_dims, in_dims)`)
  - `init_bias`: initializer for the bias vector (ignored if `bias=false`)
  - `bias`: whether to include a bias vector

## Input

  - `x` must be an Array of size `(dims..., B)` or `(dims...[0], ..., dims[k])`
    for `k ≤ size(dims)`

## Returns

  - Array of size `(dims..., B)` or `(dims...[0], ..., dims[k])` for `k ≤ size(dims)`
  - Empty `NamedTuple()`

## Parameters

  - `weight`: Weight Array of size `(dims...)`
  - `bias`: Bias of size `(dims...)`

!!! compat "Lux 0.4.3"
    
    `Scale` with multiple dimensions requires at least Lux 0.4.3.
"""
struct Scale{bias, F1, D, F2, F3} <: AbstractExplicitLayer
    activation::F1
    dims::D
    init_weight::F2
    init_bias::F3
end

function Base.show(io::IO, d::Scale)
    print(io, "Scale($(d.dims)")
    (d.activation == identity) || print(io, ", $(d.activation)")
    return print(io, ")")
end

function Scale(dims::Tuple{Vararg{Integer}}, activation=identity;
               init_weight=glorot_uniform, init_bias=zeros32, bias::Bool=true)
    activation = NNlib.fast_act(activation)
    return Scale{bias, typeof(activation), typeof(dims), typeof(init_weight),
                 typeof(init_bias)}(activation, dims, init_weight, init_bias)
end

function Scale(s1::Integer, s23::Integer...; _act=identity, kw...)
    return Scale(tuple(s1, s23...), _act; kw...)
end
Scale(size_act...; kw...) = Scale(size_act[1:(end - 1)]...; _act=size_act[end], kw...)

function initialparameters(rng::AbstractRNG, d::Scale{true})
    return (weight=d.init_weight(rng, d.dims...), bias=d.init_bias(rng, d.dims...))
end
function initialparameters(rng::AbstractRNG, d::Scale{false})
    return (weight=d.init_weight(rng, d.dims...),)
end

parameterlength(d::Scale{bias}) where {bias} = (1 + bias) * prod(d.dims)
statelength(d::Scale) = 0

function (d::Scale{true})(x::AbstractArray, ps, st::NamedTuple)
    return applyactivation(d.activation,
                           elementwise_add(elementwise_mul(ps.weight, x), ps.bias)), st
end

function (d::Scale{true, typeof(identity)})(x::AbstractArray, ps, st::NamedTuple)
    return elementwise_add(elementwise_mul(ps.weight, x), ps.bias), st
end

function (d::Scale{false})(x::AbstractArray, ps, st::NamedTuple)
    return applyactivation(d.activation, elementwise_mul(ps.weight, x)), st
end

function (d::Scale{false, typeof(identity)})(x::AbstractArray, ps, st::NamedTuple)
    return elementwise_mul(ps.weight, x), st
end
