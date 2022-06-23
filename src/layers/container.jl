"""
    SkipConnection(layer, connection)

Create a skip connection which consists of a layer or [`Chain`](@ref) of consecutive layers
and a shortcut connection linking the block's input to the output through a user-supplied
2-argument callable. The first argument to the callable will be propagated through the given
`layer` while the second is the unchanged, "skipped" input.

The simplest "ResNet"-type connection is just `SkipConnection(layer, +)`.

## Arguments

  - `layer`: Layer or `Chain` of layers to be applied to the input
  - `connection`: A 2-argument function that takes `layer(input)` and the input

## Inputs

  - `x`: Will be passed directly to `layer`

## Returns

  - Output of `connection(layer(input), input)`
  - Updated state of `layer`

## Parameters

  - Parameters of `layer`

## States

  - States of `layer`

See [`Parallel`](@ref) for a more general implementation.
"""
struct SkipConnection{T <: AbstractExplicitLayer, F} <:
       AbstractExplicitContainerLayer{(:layers,)}
    layers::T
    connection::F
end

@inline function (skip::SkipConnection)(x, ps, st::NamedTuple)
    mx, st = skip.layers(x, ps, st)
    return skip.connection(mx, x), st
end

"""
    Parallel(connection, layers...)

Create a layer which passes an input to each path in `layers`, before reducing the output
with `connection`.

## Arguments

  - `layers`: A list of `N` Lux layers
  - `connection`: An `N`-argument function that is called after passing the input through
    each layer. If `connection = nothing`, we return a tuple
    `Parallel(nothing, f, g)(x, y) = (f(x), g(y))`

## Inputs

  - `x`: If `x` is not a tuple, then return is computed as
    `connection([l(x) for l in layers]...)`. Else one is passed to each layer, thus
    `Parallel(+, f, g)(x, y) = f(x) + g(y)`.

## Returns

  - See the Inputs section for how the output is computed
  - Updated state of the `layers`

## Parameters

  - Parameters of each `layer` wrapped in a NamedTuple with
    `fields = layer_1, layer_2, ..., layer_N`

## States

  - States of each `layer` wrapped in a NamedTuple with
    `fields = layer_1, layer_2, ..., layer_N`

See also [`SkipConnection`](@ref) which is `Parallel` with one identity.
"""
struct Parallel{F, T <: NamedTuple} <: AbstractExplicitContainerLayer{(:layers,)}
    connection::F
    layers::T
end

function Parallel(connection, layers...)
    names = ntuple(i -> Symbol("layer_$i"), length(layers))
    return Parallel(connection, NamedTuple{names}(layers))
end

function (m::Parallel)(x, ps, st::NamedTuple)
    return applyparallel(m.layers, m.connection, x, ps, st)
end

Base.keys(m::Parallel) = Base.keys(getfield(m, :layers))

"""
    BranchLayer(layers...)

Takes an input `x` and passes it through all the `layers` and returns a tuple of the
outputs.

## Arguments

  - `layers`: A list of `N` Lux layers

## Inputs

  - `x`: Will be directly passed to each of the `layers`

## Returns

  - Tuple: `(layer_1(x), layer_2(x), ..., layer_N(x))`
  - Updated state of the `layers`

## Parameters

  - Parameters of each `layer` wrapped in a NamedTuple with
    `fields = layer_1, layer_2, ..., layer_N`

## States

  - States of each `layer` wrapped in a NamedTuple with
    `fields = layer_1, layer_2, ..., layer_N`

## Comparison with [`Parallel`](@ref)

This is slightly different from `Parallel(nothing, layers...)`

  - If the input is a tuple, `Parallel` will pass each element individually to each layer

  - `BranchLayer` essentially assumes 1 input comes in and is branched out into `N` outputs

## Example

An easy way to replicate an input to an NTuple is to do

```julia
l = BranchLayer(NoOpLayer(), NoOpLayer(), NoOpLayer())
```
"""
struct BranchLayer{T <: NamedTuple} <: AbstractExplicitContainerLayer{(:layers,)}
    layers::T
end

function BranchLayer(layers...)
    names = ntuple(i -> Symbol("layer_$i"), length(layers))
    return BranchLayer(NamedTuple{names}(layers))
end

function (m::BranchLayer)(x, ps, st::NamedTuple)
    return applybranching(m.layers, x, ps, st)
end

Base.keys(m::BranchLayer) = Base.keys(getfield(m, :layers))

"""
    PairwiseFusion(connection, layers...)

```
x1 → layer1 → y1 ↘
                  connection → layer2 → y2 ↘
              x2 ↗                          connection → layer3 → y3
                                        x3 ↗
```

## Arguments

  - `connection`: Takes 2 inputs and combines them
  - `layers`: [`AbstractExplicitLayer`](@ref)s

## Inputs

Layer behaves differently based on input type:

 1. If the input `x` is a tuple of length `N + 1`, then the `layers` must be a tuple of
    length `N`. The computation is as follows

```julia
y = x[1]
for i in 1:N
    y = connection(x[i + 1], layers[i](y))
end
```

 2. Any other kind of input

```julia
y = x
for i in 1:N
    y = connection(x, layers[i](y))
end
```

## Returns

  - See Inputs section for how the return value is computed
  - Updated model state for all the contained layers

## Parameters

  - Parameters of each `layer` wrapped in a NamedTuple with
    `fields = layer_1, layer_2, ..., layer_N`

## States

  - States of each `layer` wrapped in a NamedTuple with
    `fields = layer_1, layer_2, ..., layer_N`
"""
struct PairwiseFusion{F, T <: NamedTuple} <: AbstractExplicitContainerLayer{(:layers,)}
    connection::F
    layers::T
end

function PairwiseFusion(connection, layers...)
    names = ntuple(i -> Symbol("layer_$i"), length(layers))
    return PairwiseFusion(connection, NamedTuple{names}(layers))
end

function (m::PairwiseFusion)(x, ps, st::NamedTuple)
    return applypairwisefusion(m.layers, m.connection, x, ps, st)
end

Base.keys(m::PairwiseFusion) = Base.keys(getfield(m, :layers))

"""
    Chain(layers...; disable_optimizations::Bool = false)

Collects multiple layers / functions to be called in sequence on a given input.

## Arguments

  - `layers`: A list of `N` Lux layers

## Keyword Arguments

  - `disable_optimizations`: Prevents any structural optimization

## Inputs

Input `x` is passed sequentially to each layer, and must conform to the input requirements
of the internal layers.

## Returns

  - Output after sequentially applying all the layers to `x`
  - Updated model states

## Parameters

  - Parameters of each `layer` wrapped in a NamedTuple with
    `fields = layer_1, layer_2, ..., layer_N`

## States

  - States of each `layer` wrapped in a NamedTuple with
    `fields = layer_1, layer_2, ..., layer_N`

## Optimizations

Performs a few optimizations to generate reasonable architectures. Can be disabled using
keyword argument `disable_optimizations`.

  - All sublayers are recursively optimized.
  - If a function `f` is passed as a layer and it doesn't take 3 inputs, it is converted to
    a WrappedFunction(`f`) which takes only one input.
  - If the layer is a Chain, it is flattened.
  - [`NoOpLayer`](@ref)s are removed.
  - If there is only 1 layer (left after optimizations), then it is returned without the
    `Chain` wrapper.
  - If there are no layers (left after optimizations), a [`NoOpLayer`](@ref) is returned.

## Example

```julia
c = Chain(Dense(2, 3, relu), BatchNorm(3), Dense(3, 2))
```
"""
struct Chain{T} <: AbstractExplicitContainerLayer{(:layers,)}
    layers::T
    function Chain(xs...; disable_optimizations::Bool=false)
        xs = disable_optimizations ? xs : flatten_model(xs)
        length(xs) == 0 && return NoOpLayer()
        length(xs) == 1 && return first(xs)
        names = ntuple(i -> Symbol("layer_$i"), length(xs))
        layers = NamedTuple{names}(xs)
        return new{typeof(layers)}(layers)
    end
    function Chain(xs::AbstractVector; disable_optimizations::Bool=false)
        return Chain(xs...; disable_optimizations)
    end
end

function flatten_model(layers::Union{AbstractVector, Tuple})
    new_layers = []
    for l in layers
        f = flatten_model(l)
        if f isa Tuple || f isa AbstractVector
            append!(new_layers, f)
        elseif f isa Function
            if !hasmethod(f, (Any, Union{ComponentArray, NamedTuple}, NamedTuple))
                if f === identity
                    continue
                else
                    push!(new_layers, WrappedFunction(f))
                end
            else
                push!(new_layers, f)
            end
        elseif f isa Chain
            append!(new_layers, f.layers)
        elseif f isa NoOpLayer
            continue
        else
            push!(new_layers, f)
        end
    end
    return layers isa AbstractVector ? new_layers : Tuple(new_layers)
end

flatten_model(x) = x

function (c::Chain)(x, ps, st::NamedTuple)
    return applychain(c.layers, x, ps, st)
end

Base.keys(m::Chain) = Base.keys(getfield(m, :layers))
