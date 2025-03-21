
import random


def predict_seed() -> int:
    return None


def generate(seed: int) -> int:
    if seed is None or seed == 0 or seed == -1:
        seed = random.randint(0, 2**32 - 1)
        print(f"Random seed set to: {seed}")
    else:
        print(f"Seed set to: {seed}")
    return seed
