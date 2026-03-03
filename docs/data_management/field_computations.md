---
title: Implicit fields
---

`FieldGenerator` objects are given to a `PointSampler` object to be called on all the sampled points.

## Example

```python
import implicitlab as IL

sampler = PointSampler(
    geometry, # some loaded geometry
    IL.sampling_strategy.UniformBox(geometry), # the sampling strategy
    IL.fields.Occupancy(geometry, v_in=-1, v_out=1, v_on=-1) # the field to compute
)
points, field_values = sampler.sampler(10_000) 
```

This example wil sample 10k points uniformly in a bounding box around the geometry object. It returns the points and an occupancy value for each point.

## Available fields

:::implicitlab.data.fields.distance
    options:
        heading_level: 3

:::implicitlab.data.fields.occupancy
    options:
        heading_level: 3

:::implicitlab.data.fields.winding_number
    options:
        heading_level: 3

:::implicitlab.data.fields.nearest
    options:
        heading_level: 3

:::implicitlab.data.fields.misc
    options:
        heading_level: 3

## Make your custom field

The list of possible fields can be expanded by writing a custom class that inherits from the base abstract class `FieldGenerator`.

:::implicitlab.data.fields.base
    options:
        heading_level: 3

The custom class only needs to define the `compute` method. Additionnally, the `compute_on` method can be provided.