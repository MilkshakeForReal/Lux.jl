# Handling ComponentArrays
# NOTE(@avik-pal): We should probably upsteam some of these
Base.vec(c::ComponentArray) = getdata(c)

function Base.similar(c::ComponentArray, l::Vararg{Union{Integer, AbstractUnitRange}})
    return similar(getdata(c), l)
end

function Functors.functor(::Type{<:ComponentArray}, c)
    return NamedTuple{propertynames(c)}(getproperty.((c,), propertynames(c))),
           ComponentArray
end

function ComponentArrays.make_carray_args(nt::NamedTuple)
    data, ax = ComponentArrays.make_carray_args(Vector, nt)
    data = length(data) == 0 ? Float32[] :
           (length(data) == 1 ? [data[1]] : data) # no need to rely on repeat for this
    return (data, ax)
end

## For being able to print empty ComponentArrays
function ComponentArrays.last_index(f::FlatAxis)
    nt = ComponentArrays.indexmap(f)
    length(nt) == 0 && return 0
    return ComponentArrays.last_index(last(nt))
end

ComponentArrays.recursive_length(nt::NamedTuple{(), Tuple{}}) = 0

# GPU support
GPUArrays.backend(x::ComponentArray) = GPUArrays.backend(getdata(x))

function adapt_structure(to, ca::ComponentArray)
    return ComponentArray(adapt_structure(to, getdata(ca)), getaxes(ca))
end

function Base.mapreduce(f, op, A::ComponentArray{T, N, <:GPUArrays.AnyGPUArray{T, N}},
    As::GPUArrays.AbstractArrayOrBroadcasted...; dims=:,
    init=nothing) where {T, N}
    return mapreduce(f, op, getdata(A), As...; dims=dims, init=init)
end

# Seep up
function ComponentArrays.make_carray_args(A::Type{<:Vector}, nt)
    T = recursive_type(nt) # ComponentArrays don't support mixed precisions, so choose the highest
    if eltype(nt) <: AbstractGPUArray
        CUDA.@allowscalar data, idx = ComponentArrays.make_idx(cu(T[]), nt, 0) # append! triggers scalar indexing
    else
        data, idx = ComponentArrays.make_idx(T[], nt, 0)
    end
    return (data, Axis(idx))
end

function ComponentArrays.make_idx(data, x::AbstractArray, last_val) # from 6.576868s to 0.032836s!
    append!(data, x)
    out = ComponentArrays.last_index(last_val) .+ (1:length(x))
    return (data, ViewAxis(out, ShapedAxis(size(x))))
end

recursive_type(nt::NamedTuple) = mapreduce(recursive_type, promote_type, nt)
recursive_type(x::Number) = typeof(x)
recursive_type(x::AbstractArray{T,N}) where {T<:Number, N} = eltype(x)
