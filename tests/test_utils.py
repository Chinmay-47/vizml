from vizml.utils import add_docstring
import pytest


def func_without_doc():
    pass


def func_without_doc_with_args(a, b):
    print(a, b)


def func_with_doc():
    """This is a function with docstrings."""


def func_with_doc_with_args(a, b):
    """This is a function with docstrings."""
    print(a, b)


@pytest.mark.parametrize(
    "input_func", [func_without_doc, func_without_doc_with_args, func_with_doc, func_with_doc_with_args]
)
def test_add_to_docstring(input_func):
    """Tests the docstring adder function on functions without existing docs."""

    orig_doc = str(input_func.__doc__).strip()if input_func.__doc__ is not None else ''
    doc_to_add = "Added Docs for Test"

    output_func = add_docstring(input_func, doc_to_add)

    assert output_func.__doc__ == (orig_doc + "\nAdded Docs for Test").strip()
