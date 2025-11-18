# Streaming Sketches

Sketch-based streaming algorithms for estimating counts and L1 differences between two event streams. The repo contains pedagogical implementations of:

- A classic Cauchy projection sketch (`L1DifferenceSketch`) that maintains a dense random matrix and updates it online.
- A hash-based variant (`HashCauchyL1DifferenceSketch`) that recreates matrix entries on the fly so the memory footprint depends only on the sketch dimension.
- A simple Count-Min Sketch that provides the usual `epsilon`/`delta` guarantees for point queries in a data stream.

The code favors clarity over micro-optimizations and prints intermediate values so it doubles as a learning aid.

## Repository layout

```
├── README.md              ← You are here
├── requirements.txt       ← Python dependencies
└── src
    ├── main.py            ← Small driver script with demo streams
    ├── sketching.py       ← Sketch implementations
    └── utils.py           ← Helper for pretty-printing matrices
```

## Requirements

- Python 3.10+ (tested with CPython 3.11)
- `pip` and (optionally) `python -m venv`
- Dependencies listed in `requirements.txt` (currently only NumPy)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Running the examples

The driver in `src/main.py` wires up both L1-difference sketches with tiny streams so you can see the estimator in action.

```bash
python src/main.py
```

Inside `main.py`, toggle the function calls at the bottom to switch between the dense and hash-based sketches. Each run prints the sketch parameters, intermediate projections, and the final estimate so you can compare it against the hand-computed truth.

## Using the sketches from your own code

All sketches live in `src/sketching.py`. Import the class you need and process events as they arrive.

```python
from sketching import L1DifferenceSketch, HashCauchyL1DifferenceSketch, CountMinSketch

sketch = L1DifferenceSketch(n=1_000, k=50, random_seed=7)
sketch.process_stream([
    ("red", 10),
    ("blue", 10),
    ("red", 42),
])
print(sketch.estimate_l1_distance())

hash_sketch = HashCauchyL1DifferenceSketch(n=1_000_000, k=50, seed=99)
for color, idx in stream:
    hash_sketch.process_event(color, idx)

cms = CountMinSketch(epsilon=0.01, delta=0.001)
for item in stream_of_items:
    cms.update(item)
print(cms.estimate("my-item"))
```

### Choosing parameters

- `k` controls the accuracy of the L1 difference sketches: `k = Θ(1/ε²)` yields a `(1 ± ε)` approximation with constant probability.
- `n` should match the size of the event universe; the hash-based sketch never materializes the `k × n` matrix, so it is better suited to very large universes.
- For Count-Min Sketch, `epsilon` controls the additive error (`≈ ε · total_count`) and `delta` the failure probability (`1 - δ`).

## Extending the project

Ideas for further exploration:

1. Add unit tests that feed synthetic streams with known distances.
2. Expose a CLI flag to select the sketch and verbosity level.
3. Experiment with other 1-stable distributions or different hashing strategies for the projection matrix.
