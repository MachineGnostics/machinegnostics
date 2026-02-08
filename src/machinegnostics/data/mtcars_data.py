import numpy as np


def make_mtcars_check_data(seed: int = 42):
    """
    Generates a synthetic MtCars-like dataset (Motor Trend Car Road Tests).

    Size: 32 rows, 11 variables.

    Why it's famous:
    Extracted from the 1974 Motor Trend magazine. It's the go-to for learning
    linear regression (e.g., predicting MPG based on horsepower and weight).

    This function provides a reproducible, MtCars-shaped numeric dataset with
    the canonical column names and realistic relationships:
      - `mpg` negatively correlated with `hp` and `wt`.
      - `disp` correlated with `hp` and `wt`.
      - Factors like `cyl`, `vs`, `am`, `gear`, `carb` provided as integers.

    Returns
    -------
    data : numpy.ndarray
        Array of shape (32, 11) with columns in the order below.
    column_names : list[str]
        [
          'mpg','cyl','disp','hp','drat','wt','qsec','vs','am','gear','carb'
        ]
    car_names : list[str]
        Placeholder car names (Car 1..Car 32) to mimic row names.

    Notes
    -----
    - This is a synthetic, MtCars-shaped example to keep the package
      lightweight and dependency-free for this dataset.
    - Values are generated to reflect typical ranges and correlations,
      not the exact original records.

    Example
    -------
    >>> from machinegnostics.data import make_mtcars_check_data
    >>> data, cols, names = make_mtcars_check_data()
    >>> X = data[:, [3, 5]]  # hp and wt
    >>> y = data[:, 0]       # mpg
    >>> X.shape, y.shape
    ((32, 2), (32,))
    """
    rng = np.random.default_rng(seed)

    n = 32

    # Discrete engine features
    cyl_choices = np.array([4, 6, 8])
    cyl = rng.choice(cyl_choices, size=n, p=[0.4, 0.3, 0.3])

    # Base horsepower by cylinders with spread
    hp_base = np.where(cyl == 4, rng.normal(95, 15, n),
               np.where(cyl == 6, rng.normal(135, 20, n), rng.normal(190, 30, n)))
    hp = np.clip(hp_base, 50, 335)

    # Weight (in 1000 lbs), positively related to cylinders and hp
    wt = (np.where(cyl == 4, rng.normal(2.3, 0.25, n),
          np.where(cyl == 6, rng.normal(3.1, 0.30, n), rng.normal(3.8, 0.35, n))))
    wt = np.clip(wt, 1.5, 5.5)

    # Displacement, scaled with hp and wt
    disp =  (hp * rng.normal(2.0, 0.15, n)) + (wt * 50) + rng.normal(0, 20, n)
    disp = np.clip(disp, 70, 500)

    # Rear axle ratio (drat) typical range ~ 2.8 - 4.5
    drat = rng.normal(3.6 - 0.1 * (wt - 3.0), 0.2, n)
    drat = np.clip(drat, 2.5, 4.8)

    # Quarter-mile time (qsec), slower for heavier / more hp muscle cars
    qsec = rng.normal(18.0 + 0.8 * (wt - 3.0) - 0.004 * (hp - 150), 0.6, n)
    qsec = np.clip(qsec, 14.0, 23.0)

    # Engine shape and transmission
    vs = rng.integers(0, 2, size=n)  # 0 = V-shaped, 1 = straight
    am = rng.integers(0, 2, size=n)  # 0 = automatic, 1 = manual

    # Gears and carbs, loosely related to performance tier
    gear = np.where(hp < 120, rng.choice([3, 4], size=n, p=[0.6, 0.4]),
            np.where(hp < 180, rng.choice([3, 4, 5], size=n, p=[0.4, 0.45, 0.15]),
                                rng.choice([3, 4, 5], size=n, p=[0.2, 0.5, 0.3])))
    carb = np.where(hp < 120, rng.choice([1, 2], size=n, p=[0.6, 0.4]),
            np.where(hp < 180, rng.choice([2, 3, 4], size=n, p=[0.4, 0.35, 0.25]),
                                rng.choice([4, 6, 8], size=n, p=[0.5, 0.3, 0.2])))

    # MPG: negative relation with hp and wt, small effects from am/drat
    mpg = (
        35.0
        - 0.03 * hp
        - 3.0 * (wt - 3.0)
        + 0.8 * (am == 1)
        + 0.6 * (drat - 3.5)
        + rng.normal(0, 1.2, n)
    )
    mpg = np.clip(mpg, 9.0, 40.0)

    column_names = ['mpg','cyl','disp','hp','drat','wt','qsec','vs','am','gear','carb']

    data = np.column_stack([
        mpg,
        cyl,
        disp,
        hp,
        drat,
        wt,
        qsec,
        vs,
        am,
        gear,
        carb
    ]).astype(float)

    car_names = [f"Car {i+1}" for i in range(n)]

    return data, column_names, car_names
