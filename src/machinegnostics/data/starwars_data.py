import numpy as np


def make_starwars_check_data(n: int = 87, seed: int = 42):
    """
    Generates a Star Wars Characters-like dataset with height, mass, and species.

    Size: 87 observations.

    Why it's famous:
    Found in R's dplyr package. Includes height, mass, and species of 87
    characters. Commonly used for basic exploration, joins, and categorical
    analysis.

    This synthetic generator returns plausible heights (cm), masses (kg), and
    species categories. Names are provided as placeholders (Character 1..n).

    Returns
    -------
    height_cm : numpy.ndarray
        Character heights in cm, shape (n,).
    mass_kg : numpy.ndarray
        Character masses in kg, shape (n,).
    species : list[str]
        Species label for each entry (e.g., 'Human', 'Droid').
    names : list[str]
        Placeholder character names.

    Notes
    -----
    - Values are synthetic and meant for example usage only.
    - Species distribution is skewed toward 'Human' to mimic the source.

    Example
    -------
    >>> from machinegnostics.data import make_starwars_check_data
    >>> h, m, s, names = make_starwars_check_data()
    >>> len(h), len(m), len(s), len(names)
    (87, 87, 87, 87)
    """
    rng = np.random.default_rng(seed)

    species_choices = [
        'Human', 'Droid', 'Wookiee', 'Gungan', 'Twi\'lek', 'Rodian', 'Hutt', 'Other'
    ]
    species_probs = [0.55, 0.08, 0.06, 0.05, 0.06, 0.05, 0.02, 0.13]

    # Base species-wise distributions for realism
    def sample_by_species(sp: str, size: int):
        if sp == 'Human':
            h = rng.normal(175, 10, size)
            m = rng.normal(78, 15, size)
        elif sp == 'Droid':
            h = rng.normal(170, 15, size)
            m = rng.normal(85, 20, size)
        elif sp == 'Wookiee':
            h = rng.normal(210, 12, size)
            m = rng.normal(130, 20, size)
        elif sp == 'Gungan':
            h = rng.normal(200, 10, size)
            m = rng.normal(95, 18, size)
        elif sp == 'Twi\'lek':
            h = rng.normal(175, 10, size)
            m = rng.normal(70, 12, size)
        elif sp == 'Rodian':
            h = rng.normal(170, 8, size)
            m = rng.normal(65, 10, size)
        elif sp == 'Hutt':
            h = rng.normal(180, 25, size)
            m = rng.normal(300, 40, size)
        else:  # Other
            h = rng.normal(185, 20, size)
            m = rng.normal(90, 25, size)
        return h, m

    # Draw species first, then sample per-species height/mass
    species = list(rng.choice(species_choices, size=n, p=species_probs))
    height = np.empty(n)
    mass = np.empty(n)

    # Batch per unique species for stable distribution
    for sp in set(species):
        idx = [i for i, s in enumerate(species) if s == sp]
        h, m = sample_by_species(sp, len(idx))
        height[idx] = h
        mass[idx] = m

    # Clamp to reasonable ranges and add small noise
    height = np.clip(height + rng.normal(0, 1.0, n), 95, 250)
    mass = np.clip(mass + rng.normal(0, 2.0, n), 25, 600)

    names = [f"Character {i+1}" for i in range(n)]

    return height.astype(float), mass.astype(float), species, names
