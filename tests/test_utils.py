from vizml.utils import add_docstring


def test_add_to_docstring():
    """Tests the docstring adder function on functions without existing docs."""

    def func_without_doc():
        pass

    doc_to_add = "Added Docs for Test"

    func_with_new_doc = add_docstring(func_without_doc, doc_to_add)

    assert func_with_new_doc.__doc__ == "Added Docs for Test"


def test_add_docstring():
    """Tests the docstring adder function on functions with existing docs."""

    def func_with_doc():
        """This is a function with docstrings."""

    doc_to_add = "Added Docs for Test"

    func_with_added_doc = add_docstring(func_with_doc, doc_to_add)

    assert func_with_added_doc.__doc__ == "This is a function with docstrings.\nAdded Docs for Test"
