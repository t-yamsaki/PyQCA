import qca


def test_package_imports() -> None:
    assert isinstance(qca.__version__, str)
    assert qca.__version__
