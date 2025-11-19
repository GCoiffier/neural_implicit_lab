---
title: Positional Encodings
---

Encodings are a fixed transformation applied to the input before being fed into a neural network

## Usage
When defining a neural model, simply use a `nn.Sequential` object to add your encoding before your neural network. Note that the encoding size and the input dimension of the network should match.

```python
model = torch.nn.Sequential(
    RandomFourierEncoding(geometry,1000),
    MultiLayerPerceptron(1000, 128, 4)
)
```

## Available encodings

:::implicitlab.training.nn.encodings
    options:
        heading_level: 3
