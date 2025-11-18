from sketching import L1DifferenceSketch, HashCauchyL1DifferenceSketch
from math import sqrt


def l1_difference_sketch_example():
    # Example parameters
    n = 6  # Universe size
    k = 10  # Number of sketch rows
    random_seed = 42

    # Create the L1 difference sketch
    sketch = L1DifferenceSketch(n=n, k=k, random_seed=random_seed)

    # Example stream of events
    stream = [
        ("red", 1),
        ("blue", 2),
        ("red", 1),
        ("blue", 1),
        ("red", 3),
        ("blue", 2),
        ("red", 4),
        ("blue", 3),
        ("red", 5),
        ("blue", 5),
    ]

    # Process the stream
    sketch.process_stream(stream)

    # Estimate the L1 distance
    estimated_distance = sketch.estimate_l1_distance()
    true_l1_distance = 4  # Manually computed for this example
    print(f"k = epsilon^-2 = {k} <=> epsilon: {sqrt(1/k):.3f}")
    print("Bounds: ")
    print(f"Lower bound: {(1 - sqrt(1/k)) * true_l1_distance:.3f}")
    print(f"Upper bound: {(1 + sqrt(1/k)) * true_l1_distance:.3f}")
    print(f"True L1 distance: {true_l1_distance}")
    print(f"Estimated L1 distance: {estimated_distance}")


def hash_cauchy_l1_difference_sketch_example():
    n = 1000  # Universe size
    k = 10  # Number of sketch rows
    sketch = HashCauchyL1DifferenceSketch(n=n, k=k, seed=42)

    # Example stream of events
    stream = [
        ("red", 10),
        ("blue", 20),
        ("red", 10),
        ("blue", 10),
        ("red", 30),
        ("blue", 20),
        ("red", 40),
        ("blue", 30),
        ("red", 50),
        ("blue", 50),
    ]

    # Process the stream
    sketch.process_stream(stream)

    # Estimate the L1 distance
    estimated_distance = sketch.estimate_l1_distance()
    true_l1_distance = 4  # Manually computed for this example
    print(f"True L1 distance: {true_l1_distance}")
    print(f"Estimated L1 distance: {estimated_distance}")


if __name__ == "__main__":
    # l1_difference_sketch_example()
    hash_cauchy_l1_difference_sketch_example()
