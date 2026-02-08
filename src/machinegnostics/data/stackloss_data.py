import numpy as np

def make_stackloss_check_data():
    """
    Retrieves the classic Stack Loss dataset (Brownlee, 1965).
    
    This dataset describes the operation of a plant for the oxidation of 
    ammonia to nitric acid. Ideally suited for demonstrating robust regression 
    as it contains well-known outliers (notably observations 1, 2, 3, and 21).

    The dataset consists of 21 samples and 3 feature variables.

    Features (X) & Target (y):
        The dataset contains 4 columns. In some contexts 'Stack Loss' is the target, 
        but here we provide the full numerical matrix.
        1. Air Flow
        2. Water Temp.
        3. Acid Conc.
        4. Stack.Loss

    Returns
    -------
    data : numpy.ndarray
        The complete data array of shape (21, 4).
    column_names : list
        The list of column names: ['Air Flow', 'Water Temp.', 'Acid Conc.', 'Stack.Loss']

    Example
    -------
    >>> from machinegnostics.data import make_stackloss_check_data
    >>> data, names = make_stackloss_check_data()
    >>> print(f"Data shape: {data.shape}")
    Data shape: (21, 4)
    """
    
    # Data columns: Air Flow, Water Temp, Acid Conc, Stack Loss
    data = np.array([
        [80, 27, 89, 42],
        [80, 27, 88, 37],
        [75, 25, 90, 37],
        [62, 24, 87, 28],
        [62, 22, 87, 18],
        [62, 23, 87, 18],
        [62, 24, 93, 19],
        [62, 24, 93, 20],
        [58, 23, 87, 15],
        [58, 18, 80, 14],
        [58, 18, 89, 14],
        [58, 17, 88, 13],
        [58, 18, 82, 11],
        [58, 19, 93, 12],
        [50, 18, 89, 8],
        [50, 18, 86, 7],
        [50, 19, 72, 8],
        [50, 19, 79, 8],
        [50, 20, 80, 9],
        [56, 20, 82, 15],
        [70, 20, 91, 15]
    ], dtype=float)

    column_names = ['Air Flow', 'Water Temp.', 'Acid Conc.', 'Stack.Loss']

    return data, column_names
