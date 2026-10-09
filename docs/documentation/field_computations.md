---
title: Implicit fields
---

As the main goal of a neural implicit is to represent a signal over space, it is often required to generate a dataset of training points where the values for this signal is known. This job is handled by `FieldGenerator` objects. They are given as arguments to a `PointSampler` object to be called on all the sampled points.

Usual fields are implemented directly into `implicitlab`.

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