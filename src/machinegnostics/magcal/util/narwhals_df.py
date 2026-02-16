"""
Narwhals-based dataframe interoperability utilities.

This module provides a lightweight decorator, `narwhalify`, to enable seamless
multi-dataframe input/output support (Pandas, Polars, cuDF, PySpark via pandas bridge, etc.)
for functions and methods that operate internally on NumPy arrays. It ensures:

- Input wrapping: Accepts native dataframe/series objects and wraps them into Narwhals.
- Internal conversion: Converts inputs to NumPy for core numerical logic.
- Output roundtrip: Converts NumPy outputs back into the original native dataframe/series type.

The decorator is designed to be non-invasive: you keep your internal logic purely in NumPy
while gaining broad dataframe compatibility at the boundaries. It supports both free functions
and bound instance methods.
"""

import narwhals as nw
import functools
import numpy as np

def narwhalify(func):
    """
    Decorate a function or method to provide multi-dataframe IO via Narwhals.

    Purpose:
    - Accept inputs from various dataframe libraries (Pandas, Polars, cuDF, etc.).
    - Convert inputs to NumPy for internal ML/analytics logic.
    - Convert NumPy outputs back to the original native type (Series/DataFrame) when applicable.

    Supported call forms:
    - Free function: `f(data, *args, **kwargs)` where `data` is a dataframe/series/array-like.
    - Bound method: `obj.f(data, *args, **kwargs)` (first positional arg is `self`).

    Input handling:
    - Accepts dataframe-like or series-like inputs from supported libraries.
    - Uses `nw.from_native(..., eager_only=True, allow_series=True)` to construct a Narwhals object.
    - Extracts NumPy via `to_numpy()` and forwards it to your function/method.

    Output handling:
    - If your function returns `None`, it is passed through unchanged.
    - If your function returns a 1D array-like (NumPy-convertible), returns a native Series.
    - If your function returns a 2D array-like, returns a native DataFrame with generic column names
      `feature_0, feature_1, ...`.
    - If the return value is not convertible to NumPy, it is passed through unchanged.

    Notes:
    - For methods, the decorator auto-detects `self` as the first positional argument and treats
      the second positional argument as the data input.
    - Index/column propagation: When reconstructing outputs, column names are generic. If you need
      index alignment or exact column propagation, perform that in your decorated function or return
      a structure that preserves labels.
    - For functions with multiple data inputs, prefer applying the decorator to helpers that take a
      single data argument, or write a specialized wrapper.

    Examples:
    - Decorating a free function:
        >>> @narwhalify
        ... def summarize(x_numpy):
        ...     return x_numpy.mean(axis=0)

    - Decorating a method:
        >>> class Model:
        ...     @narwhalify
        ...     def predict(self, X):
        ...         # X is provided as NumPy internally
        ...         return X @ self.coef_

    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        """Wrapper adding Narwhals IO around the first data-like argument.

        Supports bound methods with positional or keyword `data` (or `X`/`z`/`y`).
        """
        native_namespace = None

        # No args: nothing to wrap
        if len(args) == 0:
            return func(*args, **kwargs)

        # Bound method?
        is_method_call = hasattr(args[0], "__class__")
        if is_method_call:
            self_obj = args[0]
            # Convert positional args (excluding self)
            converted_positional = []
            for idx, arg in enumerate(args[1:]):
                if isinstance(arg, np.ndarray):
                    converted_positional.append(arg)
                else:
                    try:
                        df = nw.from_native(arg, eager_only=True, allow_series=True)
                        if native_namespace is None:
                            native_namespace = nw.get_native_namespace(df)
                        converted_positional.append(df.to_numpy())
                    except Exception:
                        converted_positional.append(arg)

            # Convert keyword args for common data keys
            new_kwargs = dict(kwargs)
            for key in list(new_kwargs.keys()):
                val = new_kwargs[key]
                if isinstance(val, np.ndarray):
                    continue
                try:
                    df = nw.from_native(val, eager_only=True, allow_series=True)
                    if native_namespace is None:
                        native_namespace = nw.get_native_namespace(df)
                    new_kwargs[key] = df.to_numpy()
                except Exception:
                    pass

            result = func(self_obj, *converted_positional, **new_kwargs)
        else:
            # Free function: treat first positional arg as data
            converted_args = []
            for arg in args:
                if isinstance(arg, np.ndarray):
                    converted_args.append(arg)
                else:
                    try:
                        df = nw.from_native(arg, eager_only=True, allow_series=True)
                        if native_namespace is None:
                            native_namespace = nw.get_native_namespace(df)
                        converted_args.append(df.to_numpy())
                    except Exception:
                        converted_args.append(arg)
            # Convert keyword args similarly
            new_kwargs = dict(kwargs)
            for key in list(new_kwargs.keys()):
                val = new_kwargs[key]
                if isinstance(val, np.ndarray):
                    continue
                try:
                    df = nw.from_native(val, eager_only=True, allow_series=True)
                    if native_namespace is None:
                        native_namespace = nw.get_native_namespace(df)
                    new_kwargs[key] = df.to_numpy()
                except Exception:
                    pass

            result = func(*converted_args, **new_kwargs)

        # Pass-through for None or non-array-like results
        if result is None:
            return None

        try:
            result_array = np.asarray(result)
        except Exception:
            return result

        # Convert result back to original native type
        # If input was NumPy or we couldn't detect a native namespace, return NumPy/scalar result
        if native_namespace is None:
            return result_array

        # Handle scalar/0-D outputs by passing through unchanged
        if np.isscalar(result) or result_array.ndim == 0:
            return result

        # 1D -> native Series
        if result_array.ndim == 1:
            output_nw = nw.from_dict({"result": result_array}, native_namespace=native_namespace).get_column("result")
            return nw.to_native(output_nw)

        # 2D -> native DataFrame
        if result_array.ndim == 2:
            colnames = [f"feature_{i}" for i in range(result_array.shape[1])]
            output_nw = nw.from_dict(
                {col: result_array[:, i] for i, col in enumerate(colnames)},
                native_namespace=native_namespace,
            )
            return nw.to_native(output_nw)

        # Higher-dimensional outputs: return NumPy array unchanged
        return result_array

    return wrapper