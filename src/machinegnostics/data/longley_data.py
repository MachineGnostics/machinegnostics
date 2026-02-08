import numpy as np


def make_longley_check_data(seed: int = 42):
    """
    Generates a Longley-like economic dataset with high collinearity.

    Size: 16 observations.

    Why it's famous:
    A highly collinear dataset used to test the accuracy of least-squares
    regression and numerical stability.

    This synthetic generator preserves the core characteristic of the original
    dataset: strong multicollinearity among predictors such as `GNP`,
    `Population`, and time (`Year`). Values are scaled to plausible ranges.

    Columns
    -------
    ['GNP.deflator', 'GNP', 'Unemployed', 'Armed.Forces', 'Population', 'Year', 'Employed']

    Returns
    -------
    data : numpy.ndarray
        Array of shape (16, 7) with the columns listed above.
    column_names : list[str]
        Column names in order.

    Example
    -------
    >>> from machinegnostics.data import make_longley_check_data
    >>> data, cols = make_longley_check_data()
    >>> data.shape, cols
    ((16, 7), ['GNP.deflator', 'GNP', 'Unemployed', 'Armed.Forces', 'Population', 'Year', 'Employed'])
    """
    rng = np.random.default_rng(seed)

    n = 16
    year0 = 1947
    Year = np.arange(year0, year0 + n)

    # Base economic trends with smooth growth + small noise
    GNP = 200 + 12.0 * (Year - year0) + rng.normal(0, 3.5, n)
    GNP_deflator = 80 + 0.9 * (Year - year0) + rng.normal(0, 0.8, n)
    Population = 110 + 0.8 * (Year - year0) + rng.normal(0, 0.6, n)

    # Labor and defense-related stats with correlated structure
    Unemployed = 5.0 + 0.1 * (Year - year0) + 0.03 * (Population - Population.mean()) + rng.normal(0, 0.6, n)
    Armed_Forces = 3.0 + 0.05 * (Year - year0) + rng.normal(0, 0.5, n)

    # Employment as a function of GNP, Population, and Unemployed (negative coef)
    Employed = (
        50.0
        + 0.15 * (GNP - GNP.mean())
        + 0.6 * (Population - Population.mean())
        - 0.8 * (Unemployed - Unemployed.mean())
        + rng.normal(0, 0.5, n)
    )

    data = np.column_stack([
        GNP_deflator,
        GNP,
        Unemployed,
        Armed_Forces,
        Population,
        Year,
        Employed,
    ]).astype(float)

    column_names = ['GNP.deflator', 'GNP', 'Unemployed', 'Armed.Forces', 'Population', 'Year', 'Employed']

    return data, column_names
