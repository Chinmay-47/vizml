from vizml.sample import Sample


def test_sample():
    a = Sample()
    assert type(str(a)) is str
