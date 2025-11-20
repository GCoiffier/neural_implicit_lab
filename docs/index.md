---
title: ImplicitLab
---

## Overview

Implicitlab is a python library built on top of [pytorch](https://pytorch.org/) to ease the computation of _implicit neural surfaces_ (INR).

<figure markdown>
  ![](_img/representative_image_3d.jpeg){ width="800" }
</figure>

<figure markdown>
  ![](_img/representative_image_2d.jpeg){ width="400" }
</figure>


#### Fatures
- Loading, processing and sampling of various input geometry types (point clouds, polylines, surface meshes) to generate relevant training datasets  
- Training of various neural models using various published algorithms  
- Loading and saving implicit representations by serializing pytorch models  
- Perform geometrical queries and geometry processing on INRs   

## License

MIT License

Copyright (c) 2025 Guillaume Coiffier

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.