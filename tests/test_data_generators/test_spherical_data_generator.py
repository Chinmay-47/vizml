import pytest
from numpy import equal
from vizml.data_generator import SphericalDataGenerator


@pytest.mark.parametrize(
    "no_of_points", [0, 1, 2, 3]
)
def test_same_intial_generation(no_of_points):
    """New instances must generate same first values as old instances."""

    a = SphericalDataGenerator().generate(no_of_points=no_of_points)
    b = SphericalDataGenerator().generate(no_of_points=no_of_points)

    assert equal(a, b).all()


def test_setting_seed():
    """Setting seed should change the seed value."""

    gen = SphericalDataGenerator()
    gen.set_seed(new_seed=7)

    assert gen.seed_value == 7


@pytest.mark.parametrize(
    "no_of_points", [0, 1, 2, 3]
)
def test_changed_seed_changed_value(no_of_points):
    """Setting a new seed should generate new values."""

    a = SphericalDataGenerator().generate(no_of_points=no_of_points)
    gen = SphericalDataGenerator()
    gen.set_seed(new_seed=127)
    b = gen.generate(no_of_points=no_of_points)

    assert not equal(a[:, :3], b[:, :3]).any()


@pytest.mark.parametrize(
    "no_of_points", [0, 1, 2, 3]
)
def test_output_shape(no_of_points):
    """Tests the shape of the data generated."""

    a = SphericalDataGenerator().generate(no_of_points=no_of_points)
    assert a.shape == (no_of_points, 4)


def test_randomized_init_seed():
    """Initializing as randomized should set seed value as None."""

    b = SphericalDataGenerator(random=True)

    with pytest.raises(AttributeError):
        _ = b.seed_value


@pytest.mark.parametrize(
    "no_of_points", [0, 1, 2, 3]
)
def test_randomized_init_generation(no_of_points):
    """New instances must generate different first value to old instances if randomized."""

    a = SphericalDataGenerator().generate(no_of_points=no_of_points)
    b = SphericalDataGenerator(random=True).generate(no_of_points=no_of_points)

    assert not equal(a[:, :3], b[:, :3]).any()


def test_same_random_state_gen_value():
    """New instances with same random state generates same values."""

    a = SphericalDataGenerator(random_state=7).generate()
    b = SphericalDataGenerator(random_state=7).generate()

    assert equal(a, b).all()


def test_diff_random_state_gen_value():
    """New instances with different random state generates different values."""

    a = SphericalDataGenerator(random_state=7).generate()
    b = SphericalDataGenerator(random_state=11).generate()

    assert not equal(a[:, :3], b[:, :3]).any()
