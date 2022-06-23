"""
    applychain(layers::NamedTuple{fields}, x, ps, st::NamedTuple{fields})
"""
@generated function applychain(layers::NamedTuple{fields}, x, ps,
                               st::NamedTuple{fields}) where {fields}
    N = length(fields)
    x_symbols = [gensym() for _ in 1:N]
    st_symbols = [gensym() for _ in 1:N]
    calls = [
        :(($(x_symbols[1]), $(st_symbols[1])) = (layers.$(fields[1]))(x, ps.$(fields[1]),
                                                                      st.$(fields[1]))),
    ]
    append!(calls,
            [:(($(x_symbols[i]), $(st_symbols[i])) = (layers.$(fields[i]))($(x_symbols[i - 1]),
                                                                           ps.$(fields[i]),
                                                                           st.$(fields[i])))
             for i in 2:N])
    push!(calls, :(st = NamedTuple{$fields}((($(Tuple(st_symbols)...),)))))
    push!(calls, :(return $(x_symbols[N]), st))
    return Expr(:block, calls...)
end

@generated function applypairwisefusion(layers::NamedTuple{names}, connection::C, x::T, ps,
                                        st::NamedTuple) where {names, C, T}
    N = length(names)
    y_symbols = [gensym() for _ in 1:(N + 1)]
    st_symbols = [gensym() for _ in 1:N]
    getinput(i) = T <: Tuple ? :(x[$i]) : :x
    calls = [:($(y_symbols[N + 1]) = $(getinput(1)))]
    append!(calls,
            [:(($(y_symbols[i]), $(st_symbols[i])) = layers[$i]($(y_symbols[N + 1]),
                                                                ps.$(names[i]),
                                                                st.$(names[i]));
               $(y_symbols[N + 1]) = connection($(y_symbols[i]), $(getinput(i + 1))))
             for i in 1:N])
    push!(calls, :(st = NamedTuple{$names}((($(Tuple(st_symbols)...),)))))
    push!(calls, :(return $(y_symbols[N + 1]), st))
    return Expr(:block, calls...)
end

@generated function applybranching(layers::NamedTuple{names}, x, ps,
                                   st::NamedTuple) where {names}
    N = length(names)
    y_symbols = [gensym() for _ in 1:N]
    st_symbols = [gensym() for _ in 1:N]
    calls = []
    append!(calls,
            [:(($(y_symbols[i]), $(st_symbols[i])) = layers[$i](x, ps.$(names[i]),
                                                                st.$(names[i])))
             for i in 1:N])
    push!(calls, :(st = NamedTuple{$names}((($(Tuple(st_symbols)...),)))))
    push!(calls, :(return tuple($(Tuple(y_symbols)...)), st))
    return Expr(:block, calls...)
end

@generated function applyparallel(layers::NamedTuple{names}, connection::C, x::T, ps,
                                  st::NamedTuple) where {names, C, T}
    N = length(names)
    y_symbols = [gensym() for _ in 1:(N + 1)]
    st_symbols = [gensym() for _ in 1:N]
    getinput(i) = T <: Tuple ? :(x[$i]) : :x
    calls = []
    append!(calls,
            [:(($(y_symbols[i]), $(st_symbols[i])) = layers[$i]($(getinput(i)),
                                                                ps.$(names[i]),
                                                                st.$(names[i])))
             for
             i in 1:N])
    push!(calls, :(st = NamedTuple{$names}((($(Tuple(st_symbols)...),)))))
    if C == Nothing
        push!(calls, :($(y_symbols[N + 1]) = tuple($(Tuple(y_symbols[1:N])...))))
    else
        push!(calls, :($(y_symbols[N + 1]) = connection($(Tuple(y_symbols[1:N])...))))
    end
    push!(calls, :(return $(y_symbols[N + 1]), st))
    return Expr(:block, calls...)
end