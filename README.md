
The _neural implicit lab_ is a helper library built on top of pytorch for training nimplicit neural representations (INR) of geometrical objects. The goal of the library is to make the computation of an INR the ressemble a config file as much as possible and abstract away the data sampling, the training loop and the model definition, while remaining highly customizable.

**The library is under construction**. Expect heavy changes.

## Conventions

- All input geometries are centered around zero and scaled down so that they fit inside `[-1, 1]^n`
- By default, datasets of points are sampler inside the `[-1.5, 1.5]^n` domain 


## TODO list
[] Rely on `meshio` for faster geometry loading
[] Implement SAL and SALD
[] Setup the query module
[] Serialization of neural models
[] Find a better name for the lib :)