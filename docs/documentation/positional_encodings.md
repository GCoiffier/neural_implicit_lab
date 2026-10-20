---
title: Positional Encodings
---

Positional encodings are fixed transformations applied to the input data before being fed into a neural network. They increase the model's accuracy by tackling low-frequency biases. When defining a neural model, simply use a `IL.nn.Implicitequential` object to add your encoding before your neural network. Note that the encoding size and the input dimension of the network should match:

```python
# Add an encoding made of Fourier Features to a simple multi-layer perceptron
model = IL.nn.ImplicitSequential(
    RandomFourierEncoding(geometry,1000),
    MultiLayerPerceptron(1000, 128, 4)
)
```

:::implicitlab.nn.encodings.encodings
    options:
        heading_level: 2
        inherited_members: false


:::implicitlab.nn.encodings.hashgrid
    options:
        heading_level: 2
        inherited_members: false
        members:
            - MultiResDenseGridEncoding
            - MultiResHashGridEncoding