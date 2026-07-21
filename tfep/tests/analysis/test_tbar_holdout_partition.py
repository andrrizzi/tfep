import numpy as np

from tfep.analysis.tbar_holdout import choose_time_stratified_partitions


def test_time_stratified_four_way_partition_is_exact_disjoint_and_deterministic():
    counts = {
        "train": 9000,
        "small_validation": 1000,
        "large_validation": 5000,
        "reserve": 5000,
    }
    first = choose_time_stratified_partitions(20000, counts, seed=2123)
    second = choose_time_stratified_partitions(20000, counts, seed=2123)

    assert {key: len(value) for key, value in first.items()} == counts
    assert all(np.array_equal(first[key], second[key]) for key in counts)
    combined = np.concatenate([first[key] for key in counts])
    assert np.array_equal(np.sort(combined), np.arange(20000))
    assert len(np.unique(combined)) == 20000

    # Every 200-frame temporal stratum contains all four partitions.
    for start in range(0, 20000, 200):
        assert all(np.any((indices >= start) & (indices < start + 200)) for indices in first.values())
