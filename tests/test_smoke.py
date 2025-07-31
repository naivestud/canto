import canto


def test_version_is_a_string():
    assert isinstance(canto.__version__, str)
    assert canto.__version__.count(".") == 2


def test_import_is_lightweight():
    # Importing the package should not require torch model construction.
    import importlib

    module = importlib.import_module("canto")
    assert module is canto
