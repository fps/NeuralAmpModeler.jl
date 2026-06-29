module A1

    using ..Wavenet

    import Flux

    struct Layer{D, IM, L, A}
        dilated::D
        input_mixer::IM
        linear::L
        activation::A
    end

    Flux.@layer Layer

    function(m::Layer)(x, bottom_input)
        x2 = m.dilated(x)
        x2 = Wavenet.offsetadd(x2, m.input_mixer(Wavenet.valid(bottom_input, x2)))
        x_head_out = m.activation.(x2)
        x2 = m.linear(x_head_out)
        Wavenet.offsetadd(x2, x), x_head_out
    end

    struct Block{IR, L, HR}
        input_rechannel::IR
        layers::L
        head_rechannel::HR
    end

    Flux.@layer Block

    function (m::Block)(x, bottom_input, head)
        x = m.input_rechannel(x)

        for n in 1:length(m.layers)
            x, layer_head = m.layers[n](x, bottom_input)
            if isnothing(head)
                head = layer_head
            else
                head = Wavenet.offsetadd(head, layer_head)
            end
        end

        x, m.head_rechannel(Wavenet.valid(head, x))
    end

    struct Model{B, H}
        blocks::B
        head_scale::H
    end

    Flux.@layer Model

    function(m::Model)(x)
        # See: https://github.com/FluxML/Zygote.jl/issues/1261
        bottom_input = identity(x)

        head = nothing

        for block in m.blocks
            x, head = block(x, bottom_input, head)
        end

        m.head_scale .* head
    end

    Standard() = Model(
        [
            Block(
                Flux.Conv((1,), 1=>16, bias=false),
                [
                    Layer(
                        Flux.Conv((3,), 16=>16, dilation=2^k),
                        Flux.Conv((1,), 1=>16, bias=false),
                        Flux.Conv((1,), 16=>16), Flux.tanh
                    ) for k in 0:9
                ],
                Flux.Conv((1,), 16=>8, bias=false)
            ),
            Block(
                Flux.Conv((1,), 16=>8, bias=false),
                [
                    Layer(
                        Flux.Conv((3,), 8=>8, dilation=2^k),
                        Flux.Conv((1,), 1=>8, bias=false),
                        Flux.Conv((1,), 8=>8), Flux.tanh
                    ) for k in 0:9
                ],
                Flux.Conv((1,), 8=>1 )
            ),
        ],
        [0.02]
    )

    offset(m::Layer) = offset(m.dilated)

    offset(m::Block) = sum(offset, m.layers)

    offset(m::Model) = sum(offset, m.blocks)

    function set_weights(m::Flux.Conv{P, Q, R, S, T} where {P, Q, R, S, T}, weights, index)
        size_w = size(m.weight)

        new_weights = weights[index:(index+prod(size_w)-1)]

        new_w = similar(m.weight, size_w)
        index2 = 0
        for out_channel in 1:size_w[3]
            for in_channel in 1:size_w[2]
                for k in size_w[1]:-1:1
                    new_w[k, in_channel, out_channel] = weights[index + index2]
                    index2 += 1
                end
            end
        end

        m.weight .= new_w
        index += length(m.weight)

        if (m.bias != false)
            size_bias = prod(size(m.bias))
            new_bias = weights[index:(index+size_bias-1)]
            m.bias .= new_bias
            index += length(new_bias)
        end

        index
    end

    function set_weights(m::Layer, weights, index)
        index = set_weights(m.dilated, weights, index)
        index = set_weights(m.input_mixer, weights, index)
        index = set_weights(m.linear, weights, index)

        index
    end

    function set_weights(m::Block, weights, index)
        index = set_weights(m.input_rechannel, weights, index)

        for layer in m.layers
            index = set_weights(layer, weights, index)
        end

        index = set_weights(m.head_rechannel, weights, index)
    end

    function set_weights(m::Model, weights)
        index = 1
        for block in m.blocks
            index = set_weights(block, weights, index)
        end
        m.head_scale .= weights[index:index]
        index += 1
    end

end

