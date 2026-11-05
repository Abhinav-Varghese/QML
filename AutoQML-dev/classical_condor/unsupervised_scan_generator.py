#!/usr/bin/env python3
"""
Generate encoder shapes for the QUART AE architecture scan.

Fixed input size N=28 (7 constituents x 4 features each: eta, pt, cosPhi, sinPhi).
For each bottleneck B in B_VALUES (real neuron counts, already at full scale),
scans over depth (number of hidden layers, k=0..MAX_DEPTH), interpolating
hidden-layer widths evenly between N and B directly at real scale (i.e. the
interpolation itself runs between 28 and B, not between smaller qumode-scale
values that get scaled up afterward). Widths are snapped to the nearest 1/3
(rather than rounded to the nearest int), so depths beyond 3 don't collapse
into repeated widths.

e.g. B=3
  k=0: 28,3
  k=1: 28,15.33,3
  k=2: 28,19.67,11.33,3
  ...

Each shape is written as a comma-separated line to scan_shapes.txt.
"""

import numpy as np

# This N should be 3 x quantum wires
N = 21

B_VALUES = [1,2,3,4,5,6,7,8,9] # includes same compression ratios

MAX_DEPTH = 8  # k = 0..8 hidden layers

count = 0
with open("classical_condor/scan_shapes.txt", "w") as f:
    for B in B_VALUES:
        for k in range(MAX_DEPTH + 1):
            # k+2 points total: input, k hidden layers, bottleneck
            points = np.linspace(N, B, k + 2)
            # snap each point to the nearest whole integer
            shape = [int(round(x)) for x in points]
 
            f.write(",".join(str(x) for x in shape) + "\n")
            count += 1
 
print(f"Done - {count} shapes written to scan_shapes.txt")