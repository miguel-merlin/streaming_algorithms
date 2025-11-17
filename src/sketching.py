import numpy as np
from typing import Iterable, Tuple
import hashlib
import math
from utils import print_matrix_journal_style


class L1DifferenceSketch:
    """
    Sketch-based estimator for L1 distance between two streams.
    We observe a stream of items of the form (color, index), where:
        - color is 'red' or 'blue'
        - index is an integer in [0, n - 1]
    Let f_i = # of times red index i appears
        g_i = # of times blue index i appears
    We estimate D = sum_i |f_i - g_i| using a Cauchy random projection sketch.
    """

    def __init__(self, n: int, k: int, random_seed: int) -> None:
        """
        Args:
            n: Universe size (maximum index + 1)
            k: Number of sketch rows (O(1/epsilon^2) for (1 +/- epsilon) approximation)
            random_seed: Seed for random number generator
        """
        self.n = n
        self.k = k
        self.rng = np.random.default_rng(random_seed)

        # A is k x n with i.i.d  Cauchy entries
        # A_ij ~ Cauchy(0, 1)
        self.A = self.rng.standard_cauchy(size=(k, n))
        print("Random projection matrix A:")
        print_matrix_journal_style(self.A)

        # Current sketch of (f - g): t = A * (f - g)
        # We mantain t incrementally as the stream arrives
        self.t = np.zeros(k, dtype=float)
        print("Initial sketch t:")
        print(self.t)

    def process_event(self, color: str, idx: int) -> None:
        """
        Process a single event in the stream.

        Args:
            color: 'red' or 'blue'
            index: Integer index in [0, n - 1]
        """
        if not (0 <= idx < self.n):
            raise ValueError(f"Index {idx} out of bounds [0, {self.n - 1}]")
        print(f"Processing event: ({color}, {idx})")
        print(self.A[:, idx])
        if color == "red":
            self.t += self.A[:, idx]
        elif color == "blue":
            self.t -= self.A[:, idx]
        else:
            raise ValueError(f"Unknown color: {color}")
        print("Updated sketch t:")
        print(self.t)

    def estimate_l1_distance(self) -> float:
        """
        Estimate the L1 distance D = sum_i |f_i - g_i| using the current sketch.

        Returns:
            Estimated L1 distance
        """
        # The median of |t_j| over j=1..k is an unbiased estimator for D
        return float(np.median(np.abs(self.t)))

    def process_stream(self, stream: Iterable[Tuple[str, int]]) -> None:
        """
        Process a stream of events.

        Args:
            stream: Iterable of (color, index) tuples
        """
        for color, idx in stream:
            self.process_event(color, idx)


class HashCauchyL1DifferenceSketch:
    """
    Hash-based sketch for estimating the L1 distance between two streams.

    We observe a stream of (color, index) events:

        - color ∈ {"red", "blue"}
        - index ∈ {0, 1, ..., n-1}

    Let f_i = # of times red index i appears
        g_i = # of times blue index i appears

    We estimate D = sum_i |f_i - g_i| using a Cauchy 1-stable sketch,
    but without storing the full random matrix A. Instead, each A_{ij}
    is generated on the fly via a hash function.
    """

    def __init__(self, n: int, k: int, seed: int = 0):
        """
        Args:
            n: Universe size (indices are in {0, ..., n-1}).
            k: Number of sketch rows (O(1/eps^2)).
            seed: Global seed for the hash-based RNG.
        """
        if n <= 0:
            raise ValueError("n must be positive")
        if k <= 0:
            raise ValueError("k must be positive")

        self.n = n
        self.k = k
        self.seed = seed

        # Sketch vector t ∈ R^k, representing A (f - g)
        self.t = [0.0 for _ in range(k)]

    def _cauchy_entry(self, row: int, idx: int) -> float:
        """
        Deterministically generate A[row, idx] ~ Cauchy(0,1)
        using a hash of (seed, row, idx) and inverse CDF.

        Steps:
          1. Hash (seed, row, idx) -> 128-bit integer.
          2. Map integer to a uniform u ∈ (0,1).
          3. Use inverse CDF for standard Cauchy:
             X = tan(pi * (u - 1/2)).
        """
        if not (0 <= idx < self.n):
            raise ValueError(f"Index {idx} out of range [0, {self.n-1}]")

        m = hashlib.blake2b(digest_size=16)
        m.update(self.seed.to_bytes(8, "little", signed=False))
        m.update(row.to_bytes(8, "little", signed=False))
        m.update(idx.to_bytes(8, "little", signed=False))
        digest = m.digest()
        num = int.from_bytes(digest, "little")

        # Map to (0,1) (avoid exact 0 or 1 to keep tan finite)
        u = (num + 0.5) / (1 << 128)
        return math.tan(math.pi * (u - 0.5))

    def process_event(self, color: str, idx: int) -> None:
        """
        Process a single stream event (color, idx).

        - 'red' means f_idx += 1  => (f - g)_idx += 1
        - 'blue' means g_idx += 1 => (f - g)_idx -= 1
        """
        if color not in ("red", "blue"):
            raise ValueError("color must be 'red' or 'blue'")

        sign = 1.0 if color == "red" else -1.0

        # Update each sketch coordinate:
        # t[row] += sign * A[row, idx]
        for row in range(self.k):
            a_ij = self._cauchy_entry(row, idx)
            self.t[row] += sign * a_ij

    def process_stream(self, stream: Iterable[Tuple[str, int]]) -> None:
        """
        Process an entire stream of (color, idx) events.
        """
        for color, idx in stream:
            self.process_event(color, idx)

    # ---------- Query ----------

    def estimate_l1_distance(self) -> float:
        """
        Return the sketch-based estimate of D = sum_i |f_i - g_i|.

        For Cauchy 1-stable projections:
          t[row] = ⟨f - g, A_row⟩  ≈  D * Z_row,  Z_row ~ Cauchy
        and median(|Z_row|) = 1, so median(|t[row]|) ≈ D.
        """
        abs_vals = [abs(x) for x in self.t]
        abs_vals.sort()
        k = len(abs_vals)
        mid = k // 2

        if k % 2 == 1:
            return abs_vals[mid]
        else:
            return 0.5 * (abs_vals[mid - 1] + abs_vals[mid])
