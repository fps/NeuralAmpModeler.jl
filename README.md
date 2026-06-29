Only A1 standard is implemented right now.

```julia

julia> import NeuralAmpModeler

julia> m = NeuralAmpModeler.Wavenet.A1.Standard()
Model(
  [
    Block(
      Conv((1,), 1 => 16, bias=false),  # 16 parameters
      [
        Layer(
          Conv((3,), 16 => 16),         # 784 parameters
          Conv((1,), 1 => 16, bias=false),  # 16 parameters
          Conv((1,), 16 => 16),         # 272 parameters
          tanh,
        ),
        Layer(
          Conv((3,), 16 => 16, dilation=2),  # 784 parameters
          Conv((1,), 1 => 16, bias=false),  # 16 parameters
          Conv((1,), 16 => 16),         # 272 parameters
          tanh,
        ),
        Layer(
          Conv((3,), 16 => 16, dilation=4),  # 784 parameters
          Conv((1,), 1 => 16, bias=false),  # 16 parameters
          Conv((1,), 16 => 16),         # 272 parameters
          tanh,
        ),
        Layer(
          Conv((3,), 16 => 16, dilation=8),  # 784 parameters
          Conv((1,), 1 => 16, bias=false),  # 16 parameters
          Conv((1,), 16 => 16),         # 272 parameters
          tanh,
        ),
        Layer(
          Conv((3,), 16 => 16, dilation=16),  # 784 parameters
          Conv((1,), 1 => 16, bias=false),  # 16 parameters
          Conv((1,), 16 => 16),         # 272 parameters
          tanh,
        ),
        Layer(
          Conv((3,), 16 => 16, dilation=32),  # 784 parameters
          Conv((1,), 1 => 16, bias=false),  # 16 parameters
          Conv((1,), 16 => 16),         # 272 parameters
          tanh,
        ),
        Layer(
          Conv((3,), 16 => 16, dilation=64),  # 784 parameters
          Conv((1,), 1 => 16, bias=false),  # 16 parameters
          Conv((1,), 16 => 16),         # 272 parameters
          tanh,
        ),
        Layer(
          Conv((3,), 16 => 16, dilation=128),  # 784 parameters
          Conv((1,), 1 => 16, bias=false),  # 16 parameters
          Conv((1,), 16 => 16),         # 272 parameters
          tanh,
        ),
        Layer(
          Conv((3,), 16 => 16, dilation=256),  # 784 parameters
          Conv((1,), 1 => 16, bias=false),  # 16 parameters
          Conv((1,), 16 => 16),         # 272 parameters
          tanh,
        ),
        Layer(
          Conv((3,), 16 => 16, dilation=512),  # 784 parameters
          Conv((1,), 1 => 16, bias=false),  # 16 parameters
          Conv((1,), 16 => 16),         # 272 parameters
          tanh,
        ),
      ],
      Conv((1,), 16 => 8, bias=false),  # 128 parameters
    ),
    Block(
      Conv((1,), 16 => 8, bias=false),  # 128 parameters
      [
        Layer(
          Conv((3,), 8 => 8),           # 200 parameters
          Conv((1,), 1 => 8, bias=false),  # 8 parameters
          Conv((1,), 8 => 8),           # 72 parameters
          tanh,
        ),
        Layer(
          Conv((3,), 8 => 8, dilation=2),  # 200 parameters
          Conv((1,), 1 => 8, bias=false),  # 8 parameters
          Conv((1,), 8 => 8),           # 72 parameters
          tanh,
        ),
        Layer(
          Conv((3,), 8 => 8, dilation=4),  # 200 parameters
          Conv((1,), 1 => 8, bias=false),  # 8 parameters
          Conv((1,), 8 => 8),           # 72 parameters
          tanh,
        ),
        Layer(
          Conv((3,), 8 => 8, dilation=8),  # 200 parameters
          Conv((1,), 1 => 8, bias=false),  # 8 parameters
          Conv((1,), 8 => 8),           # 72 parameters
          tanh,
        ),
        Layer(
          Conv((3,), 8 => 8, dilation=16),  # 200 parameters
          Conv((1,), 1 => 8, bias=false),  # 8 parameters
          Conv((1,), 8 => 8),           # 72 parameters
          tanh,
        ),
        Layer(
          Conv((3,), 8 => 8, dilation=32),  # 200 parameters
          Conv((1,), 1 => 8, bias=false),  # 8 parameters
          Conv((1,), 8 => 8),           # 72 parameters
          tanh,
        ),
        Layer(
          Conv((3,), 8 => 8, dilation=64),  # 200 parameters
          Conv((1,), 1 => 8, bias=false),  # 8 parameters
          Conv((1,), 8 => 8),           # 72 parameters
          tanh,
        ),
        Layer(
          Conv((3,), 8 => 8, dilation=128),  # 200 parameters
          Conv((1,), 1 => 8, bias=false),  # 8 parameters
          Conv((1,), 8 => 8),           # 72 parameters
          tanh,
        ),
        Layer(
          Conv((3,), 8 => 8, dilation=256),  # 200 parameters
          Conv((1,), 1 => 8, bias=false),  # 8 parameters
          Conv((1,), 8 => 8),           # 72 parameters
          tanh,
        ),
        Layer(
          Conv((3,), 8 => 8, dilation=512),  # 200 parameters
          Conv((1,), 1 => 8, bias=false),  # 8 parameters
          Conv((1,), 8 => 8),           # 72 parameters
          tanh,
        ),
      ],
      Conv((1,), 8 => 1),               # 9 parameters
    ),
  ],
  1-element Vector{Float64},            # 1 parameters
)                   # Total: 106 arrays, 13_802 parameters, 62.723 KiB.

julia> import JSON

julia> new_weights = Float32.(JSON.parsefile("model.nam").weights)
13802-element Vector{Float32}:
  0.0020099105
  0.32737935
 -0.8225299
 -0.73060954
 -0.43824032
  0.20229235
 -0.0062850104
  0.80599976
 -0.07333748
  0.13569129
 -0.2723522
 -0.0059857965
 -0.88748103
 -0.742199
 -0.47862476
  0.0025243533
  0.14382565
  0.07608561
  0.06531755
  0.014887172
  0.098348245
  0.09959571
  ⋮
  0.3364353
 -0.22672084
  0.120582595
  0.21581474
 -0.23014374
  0.060143214
 -0.19203112
 -0.2548099
  0.34733036
 -0.28660518
 -0.32359937
  0.29442775
  1.2653815
  0.8371381
  1.1606729
  0.6211958
 -1.0078379
 -0.955654
  0.8637494
 -1.0744867
 -0.05586061
  0.02

julia> NeuralAmpModeler.Wavenet.A1.set_weights(m, new_weights);
```
