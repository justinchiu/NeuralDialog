
from more_itertools import powerset
from collections import Counter

def get_latent_powerset(goal):
    flat_goal = [y for x, c in zip(range(3), goal[::2]) for y in [x] * c]
    flat_partitions = [Counter(x) for x in powerset(flat_goal)]
    counts = Counter(flat_goal)
    flat_complements = [counts - x for x in flat_partitions]
    # first is you, second is partner
    partitions = [(x[0], x[1], x[2], y[0], y[1], y[2]) for x, y in zip(flat_partitions, flat_complements)] 
    return list(set(partitions)) + [[11] * 6]

