from typing import Callable, Any, Union


def add_docstring(input_func: Union[Callable[[], Any], Callable[[Any], Any]],
                  doc_to_add: str) -> Union[Callable[[], Any], Callable[[Any], Any]]:
    """
    Helper function to add docstrings to other functions.
    """

    existing_docstring = ['\n', str(input_func.__doc__).strip(), '\n'] if input_func.__doc__ is not None else []
    input_func.__doc__ = ''.join(existing_docstring + [doc_to_add]).strip()

    return input_func
