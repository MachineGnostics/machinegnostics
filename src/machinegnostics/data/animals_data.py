import numpy as np

def make_animals_check_data():
    """
    Retrieves the 'Animals' dataset (Rousseeuw & Leroy, 1987).
    
    A classic small and challenging dataset for robust regression. 
    It contains body weight (kg) and brain weight (g) for 28 animals.
    
    Challenge:
    1. **Scale:** The values span several orders of magnitude (Mouse vs Brachiosaurus).
    2. **Outliers:** The dataset includes 3 dinosaurs (Diplodocus, Brachiosaurus, Triceratops).
       These act as "bad leverage" points—they have massive body weights (high X) 
       but very small brains (low Y) relative to the mammalian trend.
       Standard Linear Regression will try to fit the dinosaurs and fail to model 
       the mammals correctly.
       
    Typical Usage:
    Fit log(Brain) ~ log(Body).

    Returns
    -------
    X : numpy.ndarray
        Body weight in kg. Shape (28, 1).
    y : numpy.ndarray
        Brain weight in g. Shape (28,).
    names : list of str
        The common name of the animal for each data point.

    Example
    -------
    >>> from machinegnostics.data.animals_data import make_animals_check_data
    >>> X, y, names = make_animals_check_data()
    >>> print(f"Heaviest: {names[np.argmax(X)]} ({np.max(X)} kg)")
    Heaviest: Brachiosaurus (87000.0 kg)
    """
    
    # Format: [Body Weight (kg), Brain Weight (g)]
    data = np.array([
        [1.35, 8.1],       # Mountain beaver
        [465., 423.],      # Cow
        [36.33, 119.5],    # Grey wolf
        [27.66, 115.],     # Goat
        [1.04, 5.5],       # Guinea pig
        [11700., 50.],     # Diplodocus (Dinosaur)
        [2547., 4603.],    # Asian elephant
        [6.8, 179.],       # Rhesus monkey
        [35., 56.],        # Kangaroo
        [0.12, 1.],        # Golden hamster
        [0.023, 0.4],      # Mouse
        [2.5, 12.1],       # Rabbit
        [55.5, 175.],      # Sheep
        [100., 157.],      # Jaguar
        [52.16, 440.],     # Chimpanzee
        [0.28, 1.9],       # Rat
        [87000., 154.5],   # Brachiosaurus (Dinosaur)
        [0.122, 3.],       # Mole
        [192., 180.],      # Pig
        [3., 25.],         # Echidna
        [9400., 70.],      # Triceratops (Dinosaur)
        [0.1, 4.],         # Pigmy marmoset
        [6654., 5712.],    # African elephant
        [62., 1320.],      # Human
        [10., 115.],       # Potar monkey
        [3.3, 25.6],       # Cat
        [529., 680.],      # Giraffe
        [207., 406.]       # Gorilla
    ])
    
    names = [
        "Mountain beaver", "Cow", "Grey wolf", "Goat", "Guinea pig", "Diplodocus",
        "Asian elephant", "Rhesus monkey", "Kangaroo", "Golden hamster", "Mouse",
        "Rabbit", "Sheep", "Jaguar", "Chimpanzee", "Rat", "Brachiosaurus",
        "Mole", "Pig", "Echidna", "Triceratops", "Pigmy marmoset", "African elephant",
        "Human", "Potar monkey", "Cat", "Giraffe", "Gorilla"
    ]

    X = data[:, 0:1]
    y = data[:, 1]

    return X, y, names
