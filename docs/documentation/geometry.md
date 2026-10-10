---
title: Geometrical objects
---

Geometrical objects in the implicit lab are stored using the [mouette](https://gcoiffier.github.io/mouette/) library, which are either:

- A [point cloud](https://gcoiffier.github.io/mouette/datastructures/PointClouds/)  
- A [polyline](https://gcoiffier.github.io/mouette/datastructures/PolyLines/)  
- A [surface mesh](https://gcoiffier.github.io/mouette/datastructures/SurfaceMeshes/)  


Volume meshes are not supported for now.

In addition to the capabilities of mouette classes, each geometrical object is assigned a `GeometryType` attribute:

:::implicitlab.data.geometry
    options:
        heading_level: 2
        members:
            - "GeometryType"

## Geometry utilities

:::implicitlab.data.geometry
    options:
        heading_level: 3
        filters:
            - "!GeometryType"


:::implicitlab.data.sample_utils
    options:
        heading_level: 3