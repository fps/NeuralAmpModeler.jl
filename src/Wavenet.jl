module Wavenet
    function offsetadd(x1, x2)
        if size(x1, 1) < size(x2, 1)
            x1 + x2[(end-size(x1, 1)+1):end, :, :]
        else
            x2 + x1[(end-size(x2, 1)+1):end, :, :]
        end
    end

    function valid(x, s::Integer)
        x[(end-s+1):end, :, :]
    end

    function valid(x, s)
        x[(end-size(s, 1)+1):end, :, :]
    end

    include("A1.jl")
end
