---
title: Dataset sampling
---

Creating a dataset of points with associated signal values is handled by a `PointSampler` object.

:::implicitlab.data.sampler
    options:
        heading_level: 2
        members:
            - PointSampler

## Point Sampling strategies

:::implicitlab.data.sampling_strategies
    options:
        heading_level: 3
        filters:
            - "!PointSampler"

### Creating your own custom strategy

A custom strategy can be defined by writing a custom class that inherits from the abstract base class `SamplingStrategy`:

```python
from implicitlab.data.sampling_strategies import SamplingStrategy

class MyCustomSamplingStrategy(SamplingStrategy):

    def __init__(self, *args):
        # Implement the initialization of the data needed for your strategy here
    
    def sample(self, n_pts: int) -> np.ndarray:
        # Implement your sampling here
```

## Examples of use


```python
"""
Define a point sampler for the signed distance to the geometry, with a combined strategy
"""
import implicitlab as IL
from implicilab import sampling_strategies as strats

# The defined sampling strategy : 
# 2/3 of the points are taken in a uniform box (by default [-1, 1]^d)
# around the shape. 1/3 of the points are computed as a sampling on 
# the geometry plus some Gaussian noise of standard deviation 0.1
sampling_strat = strats.CombinedStrategy([
    strats.UniformBox(geometry),
    strats.NearGeometryGaussian(geometry, stdv=0.1)
], [2., 1.])

# The signal to compute for each point is the signed distance to the geometry
field = IL.fields.Distance(geometry, signed=True, square=False)

# The sampler is created
sampler = IL.PointSampler(
    geometry, # some loaded geometry
    sampling_strat, # the strategy
    field # the field
)

# Sample 10k points
points, sdfs = sampler.sample(10_000) 
```


```python
"""
Define a point sampler with two different fields (occupancy and nearest point)"""

import implicitlab as IL
from implicilab import sampling_strategies as strats

sampler = IL.PointSampler(
    geometry, # some loaded geometry
    IL.sampling_strategy.UniformBox(geometry), # the sampling strategy
    IL.fields.Occupancy(geometry, v_in=-1, v_out=1, v_on=-1), # the occupancy field
    IL.fields.Nearest(geometry) # the nearest point field
)
points, field_values = sampler.sampler(10_000) 
```

This example wil sample 10k points uniformly in a bounding box around the geometry object. It returns the points and an occupancy value for each point.