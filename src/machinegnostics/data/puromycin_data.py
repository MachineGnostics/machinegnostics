import numpy as np


def make_puromycin_check_data(seed: int = 42):
    """
    Generates a Puromycin-like biochemical reaction rate dataset.

    Size: 23 rows.

    Why it's famous:
    Used to show biochemical reaction rates. It’s a classic for
    non-linear regression modeling (e.g., Michaelis–Menten kinetics)
    comparing treated vs untreated samples.

    This function generates two groups (state: untreated=0, treated=1),
    each following a Michaelis–Menten curve with distinct parameters and
    small Gaussian noise.

    Returns
    -------
    data : numpy.ndarray
        Array of shape (23, 3) with columns [conc, rate, state].
        `state` is 0 for untreated and 1 for treated.
    column_names : list[str]
        ['conc', 'rate', 'state']

    Example
    -------
    >>> from machinegnostics.data import make_puromycin_check_data
    >>> data, cols = make_puromycin_check_data()
    >>> data.shape, cols
    ((23, 3), ['conc', 'rate', 'state'])
    """
    rng = np.random.default_rng(seed)

    # Concentration grid, more dense at low concentrations
    conc_vals = np.r_[
        np.linspace(0.02, 0.2, 8),
        np.linspace(0.25, 1.2, 7),
        np.linspace(1.3, 2.0, 8)
    ][:23]

    # Parameters for untreated and treated groups
    # Treated usually exhibits larger Vmax or lower Km
    Vmax_u, Km_u = 115.0, 0.50
    Vmax_t, Km_t = 130.0, 0.35

    # Split across states with slight random assignment to reach 23 total
    state = np.zeros(conc_vals.size, dtype=int)
    treated_idx = rng.choice(conc_vals.size, size=11, replace=False)
    state[treated_idx] = 1

    # Michaelis–Menten curves with noise
    rate = np.empty_like(conc_vals)
    for i, (c, s) in enumerate(zip(conc_vals, state)):
        if s == 0:
            mean = Vmax_u * c / (Km_u + c)
        else:
            mean = Vmax_t * c / (Km_t + c)
        rate[i] = mean + rng.normal(0, 3.0)

    data = np.column_stack([conc_vals, rate, state]).astype(float)
    column_names = ['conc', 'rate', 'state']
    return data, column_names
