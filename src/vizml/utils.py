from typing import Callable


def add_docstring(input_func: Callable, doc_to_add: str) -> Callable:
    """
    Helper function to add docstrings to other functions.
    """

    existing_docstring = ['\n', str(input_func.__doc__).strip(), '\n'] if input_func.__doc__ is not None else []
    input_func.__doc__ = ''.join(existing_docstring + [doc_to_add]).strip()

    return input_func
