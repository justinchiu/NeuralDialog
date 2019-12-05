import itertools
import random
import pickle
from pathlib import Path

import numpy as np

save_file = Path("contexts.pkl")

if save_file.exists():
    with save_file.open("rb") as f:
        (
            contexts,
            context_to_value_functions,
            context_value_pairs_to_value_functions,
        ) = pickle.load(f)
else:
    contexts = [
        context for context in itertools.product(range(1, 6), repeat=3)
        if 5 <= sum(context) <= 7
    ]

    context_to_value_functions = {}
    for context in contexts:
        context_to_value_functions[context] = []
        for value_function in itertools.product(range(11), repeat=3):
            if np.dot(context, value_function) == 10:
                context_to_value_functions[context].append(value_function)

    context_value_pairs_to_value_functions = {}
    for context in context_to_value_functions:
        context_value_pairs_to_value_functions[context] = {}
        for first_value_function in context_to_value_functions[context]:
            context_value_pairs_to_value_functions[context][first_value_function] = []
            for second_value_function in context_to_value_functions[context]:
                # Check every item has some value assigned to it:
                sum_of_values = np.add(first_value_function, second_value_function)
                all_items_valued = np.all(sum_of_values > 0)
                # Check at least one item is valued by both players:
                some_item_valued_twice = np.any(np.logical_and(np.asarray(first_value_function) > 0, np.asarray(second_value_function) > 0))
                if all_items_valued and some_item_valued_twice:
                    context_value_pairs_to_value_functions[context][first_value_function].append(second_value_function)
    with save_file.open("wb") as f:
        pickle.dump(
            (contexts, context_to_value_functions, context_value_pairs_to_value_functions),
            f,
        )

def get_valid_contexts(ctx):
    # ctx: [6 * str(int)]
    context = tuple(int(x) for x in ctx[::2])
    values = tuple(int(x) for x in ctx[1::2])
    # [(3 * int)]
    valid_values = context_value_pairs_to_value_functions[context][values]
    # convert to [(6 * str(int))], interleaving count and values
    valid_contexts = [
        [str(x) for xs in zip(context, vs) for x in xs]
        for vs in valid_values
    ]
    return valid_contexts

def get_valid_contexts_ints(ctx):
    # ctx: [6 * str(int)]
    context = tuple(int(x) for x in ctx[::2])
    values = tuple(int(x) for x in ctx[1::2])
    # [(3 * int)]
    valid_values = context_value_pairs_to_value_functions[context][values]
    # convert to [(6 * str(int))], interleaving count and values
    valid_contexts = [
        [x for xs in zip(context, vs) for x in xs]
        for vs in valid_values
    ]
    return valid_contexts

def sample_max_contexts(x):
    # need to return str version for Denis dictionary
    str_ctxs = get_valid_contexts(x)
    ctxs = get_valid_contexts_ints(x)
    cs = np.array(ctxs)
    values = cs[:,1::2]
    maxes = values == values.max(-1, keepdims=True)
    is_empty = maxes.sum(0) == 0
    if is_empty.any():
        # one bucket is empty, just set it to uniform over all contexts.
        maxes[:,is_empty] = True
    probs = maxes / maxes.sum(0, keepdims=True)
    N = cs.shape[0]
    # sample with replacement, whatever
    chosen_contexts = [
        str_ctxs[np.random.choice(N, p=probs[:,i])]
        for i in range(3)
    ]
    return chosen_contexts

if __name__ == "__main__":
    assert len(contexts) == 31
    print(
        'Size of union of valid value functions for all contexts:',
        len(set(
            value_function
            for context in contexts
            for value_function in context_to_value_functions[context]))
    )

    for context, value_functions in sorted(context_to_value_functions.items()):
        print(context, 'has', len(value_functions), 'valid value functions')
        #print(value_functions)
        #
    print([
        sample_max_contexts([ctx[0], value[0], ctx[1], value[1], ctx[2], value[2]])
        for ctx, values in context_to_value_functions.items()
        for value in values
    ])
    import pdb; pdb.set_trace()
