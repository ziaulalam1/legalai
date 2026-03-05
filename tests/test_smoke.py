from ldc.schema import Label


def test_labels():
    # five classes, no more no less
    args = Label.__args__
    assert set(args) == {"motion", "brief", "deposition", "order", "exhibit"}
    assert len(args) == 5
