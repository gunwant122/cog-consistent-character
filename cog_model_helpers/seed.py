import random


def get_default_seed() -> int:
    """Get default seed value (None)"""
    return None


def generate(seed: int) -> int:
    """Generate a seed value. If seed is None, 0, or -1, generate a random seed."""
    if seed is None or seed == 0 or seed == -1:
        seed = random.randint(0, 2**32 - 1)
        print(f"Random seed set to: {seed}")
    else:
        print(f"Seed set to: {seed}")
    return seed
