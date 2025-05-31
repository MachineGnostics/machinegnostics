def disable_parent_docstring(func):
    """
    Decorator to disable (remove) the inherited docstring from a parent class method.
    After applying this decorator, the function's __doc__ will be set to None.
    """
    func.__doc__ = None
    return func