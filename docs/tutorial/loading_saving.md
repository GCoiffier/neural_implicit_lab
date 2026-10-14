---
title: Loading and saving implicit models
---

`implicitlab` provides a mechanism for seamlessly load and save neural models. 

## Saving a model

If your model inherits from the base `ImplicitModel` class, or is an `ImplicitSequential` object, simply call:

```python
model.save("path/to/saved/file.inr")
```

We use the `.inr` extension as a convention, but the file is actually serialized using the `torch.save` function.

## Loading

Loading from a file on the disk is also very simple:

```python
import implicitlab as IL
model = IL.nn.load_model("path/to/model.inr")
model = IL.nn.ImplicitModel.load("path/to/model.inr") # equivalent call
```


The loading will fail if the path does not correspond to a previously saved model.

## Working with custom models

As it is possible to define your own neural model using the library, making them inherit from `ImplicitModel` (or `ImplicitSequential`) will allow them to be saved and loaded back.For instance:

```python
import implicitlab as IL

class MyCustomModel(IL.nn.ImplicitModel):
    def __init__(self, ...):
        super().__init__()


mod = MyCustomModel(...)
mod.save("model.inr")
mod2 = IL.nn.load_model("model.inr")
```
will work with no issues. Loading a `MyCustomModel` object will also work from another file in another python session _as long as the path to the class definition is available from the script it is loaded in_.
