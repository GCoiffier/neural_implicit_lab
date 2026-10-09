---
title: Preparing a dataset
---

# Working with geometric data

The implicit lab supports various format of geometrical objects as input:
- point clouds (and oriented point clouds)
- polylines
- surface meshes


File format include `.xyz`, `.mesh`, `.obj`, `.stl`.


# Loading geometry

Loading some geometry object can simply be done using the `load_geometry` function:

```python
import implicitlab as IL

geom_object = IL.load_geometry("path/to/my/file.obj")
print(geom_object.geom_type)
print(geom_object.dim)
```

We rely on the datastructures from [mouette](https://github.com/GCoiffier/mouette) to store the geometry. In addition, each geometrical object has two attributes: a type and a dimension ()


# Preparing a dataset

## The PointSampler Class

