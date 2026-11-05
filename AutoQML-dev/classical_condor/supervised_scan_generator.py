#!/usr/bin/env python3

"""
Generate classical NN classifier architectures for a fixed-width
depth scan.

For a given input size N (28 = 7 constituents x 4 features each:
eta, pt, cosPhi, sinPhi):

    0 hidden layers:
        N -> 1

    For each width w in WIDTHS:

        1 hidden layer:
            N -> w -> 1

        2 hidden layers:
            N -> w -> w -> 1

        3 hidden layers:
            N -> w -> w -> w -> 1

        ...

    Thus, width is fixed within an architecture and only depth varies.

    Note: shapes here describe the encoder trunk only (input -> hidden
    widths). The final ->1 classifier head shown above is added by the
    model itself (self.classifier = nn.Linear(layer_sizes[-1], 1)) and
    must NOT be included in these shapes, or it gets double-applied.
"""

# Remember this is 3 * constituents
N = 18

# Widths as multiples of N
WIDTH_FACTORS = [0.5, 1.0, 1.5, 2.0]

# Number of hidden layers to scan
MAX_HIDDEN = 8

OUTPUT_FILE = "classical_condor/sup_scan_shapes.txt"


# ============================================================
# Generate widths
# ============================================================

WIDTHS = [
    round(N * factor)
    for factor in WIDTH_FACTORS
]


# ============================================================
# Generate shapes
# ============================================================

count = 0

with open(OUTPUT_FILE, "w") as f:

    # --------------------------------------------------------
    # 0 hidden layers
    # --------------------------------------------------------

    shape = (N,)

    f.write(",".join(str(x) for x in shape) + "\n")
    count += 1

    # --------------------------------------------------------
    # Fixed width, increasing depth
    # --------------------------------------------------------

    for width in WIDTHS:

        for depth in range(1, MAX_HIDDEN + 1):

            hidden = (width,) * depth
            shape = (N,) + hidden

            f.write(",".join(str(x) for x in shape) + "\n")
            count += 1


print(f"N = {N}")
print(f"Widths = {WIDTHS}")
print(f"Depths = 1-{MAX_HIDDEN}")
print(f"Done — {count} shapes written to {OUTPUT_FILE}")