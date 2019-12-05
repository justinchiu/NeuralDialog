
from more_itertools import powerset
from collections import Counter

def get_latent_powerset(goal):
    flat_goal = [y for x, c in zip(range(3), goal[::2]) for y in [x] * c]
    flat_partners = [Counter(x) for x in powerset(flat_goal)]
    # unflatten partner goal
    return [(x[0], x[1], x[2]) for x in flat_partners]
