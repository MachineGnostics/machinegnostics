def disable_parent_docstring(func):
    """
    Decorator to disable (remove) the inherited docstring from a parent class method.
    After applying this decorator, the function's __doc__ will be set to None.

    Usage:

    ```python
    @disable_parent_docstring
    def my_method(self, *args, **kwargs):
        # Your method implementation here
        pass
    ```    
    This is useful when you want to override a method from a parent class
    but do not want to inherit its docstring, allowing you to provide a new one or leave it empty.
    """
    func.__doc__ = None
    return func