"""
    ReshapeLayer(dims)

Reshapes the passed array to have a size of `(dims..., :)`

## Arguments

  - `dims`: The new dimensions of the array (excluding the last dimension).

## Inputs

  - `x`: AbstractArray of any shape which can be reshaped in `(dims..., size(x, ndims(x)))`

## Returns

  - AbstractArray of size `(dims..., size(x, ndims(x)))`
  - Empty `NamedTuple()`
"""
struct ReshapeLayer{N} <: AbstractExplicitLayer
    dims::NTuple{N, Int}
end

@inline function (r::ReshapeLayer)(x::AbstractArray, ps, st::NamedTuple)
    return reshape(x, r.dims..., size(x, ndims(x))), st
end

function Base.show(io::IO, r::ReshapeLayer)
    return print(io, "ReshapeLayer(output_dims = (", join(r.dims, ", "), ", :))")
end

"""
    FlattenLayer()

Flattens the passed array into a matrix.

## Inputs

  - `x`: AbstractArray

## Returns

  - AbstractMatrix of size `(:, size(x, ndims(x)))`
  - Empty `NamedTuple()`
"""
struct FlattenLayer <: AbstractExplicitLayer end

@inline function (f::FlattenLayer)(x::AbstractArray{T, N}, ps, st::NamedTuple) where {T, N}
    return reshape(x, :, size(x, N)), st
end

"""
    SelectDim(dim, i)

Return a view of all the data of the input `x` where the index for dimension `dim` equals
`i`. Equivalent to `view(x,:,:,...,i,:,:,...)` where `i` is in position `d`.

## Arguments

  - `dim`: Dimension for indexing
  - `i`: Index for dimension `dim`

## Inputs

  - `x`: AbstractArray that can be indexed with `view(x,:,:,...,i,:,:,...)`

## Returns

  - `view(x,:,:,...,i,:,:,...)` where `i` is in position `d`
  - Empty `NamedTuple()`
"""
struct SelectDim{dim, index} <: AbstractExplicitLayer end

SelectDim(dim, index) = SelectDim{Val(dim), Val(index)}()

@inline function (s::SelectDim{dim, index})(x, ps, st::NamedTuple) where {dim, index}
    return selectdim(x, get_known(dim), get_known(index)), st
end

function Base.show(io::IO, s::SelectDim{dim, index}) where {dim, index}
    return print(io, "SelectDim(dim = ", get_known(dim), ", index = ", get_known(index),
                 ")")
end

"""
    NoOpLayer()

As the name suggests does nothing but allows pretty printing of layers. Whatever input is
passed is returned.
"""
struct NoOpLayer <: AbstractExplicitLayer end

@inline (noop::NoOpLayer)(x, ps, st::NamedTuple) = x, st

"""
    WrappedFunction(f)

Wraps a stateless and parameter less function. Might be used when a function is added to
`Chain`. For example, `Chain(x -> relu.(x))` would not work and the right thing to do would
be `Chain((x, ps, st) -> (relu.(x), st))`. An easier thing to do would be
`Chain(WrappedFunction(Base.Fix1(broadcast, relu)))`

## Arguments

  - `f::Function`: A stateless and parameterless function

## Inputs

  - `x`: s.t `hasmethod(f, (typeof(x),))` is `true`

## Returns

  - Output of `f(x)`
  - Empty `NamedTuple()`
"""
struct WrappedFunction{F} <: AbstractExplicitLayer
    func::F
end

(wf::WrappedFunction)(x, ps, st::NamedTuple) = wf.func(x), st

function Base.show(io::IO, w::WrappedFunction)
    return print(io, "WrappedFunction(", w.func, ")")
end

"""
    ActivationFunction(f)

Broadcast `f` on the input but fallback to CUDNN for Backward Pass. Internally
calls [`Lux.applyactivation`](@ref)

## Arguments

  - `f`: Activation function

## Inputs

  - `x`: Any array type s.t. `f` can be broadcasted over it

## Returns

  - Broadcasted Activation `f.(x)`
  - Empty `NamedTuple()`
"""
struct ActivationFunction{F} <: AbstractExplicitLayer
    func::F
end

(af::ActivationFunction)(x, ps, st::NamedTuple) = applyactivation(af.func, x), st

function Base.show(io::IO, af::ActivationFunction)
    return print(io, "ActivationFunction(", af.func, ")")
end