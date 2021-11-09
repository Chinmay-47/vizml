import pytest
from numpy import equal
from vizml.data_generator import Normal2DGenerator


@pytest.mark.parametrize(
    "no_of_points", [0, 1, 2, 3]
)
def test_same_intial_generation(no_of_points):
    """New instances must generate same first values as old instances."""

    a = Normal2DGenerator().generate(no_of_points=no_of_points)
    b = Normal2DGenerator().generate(no_of_points=no_of_points)

    assert equal(a, b).all()


def test_setting_seed():
    """Setting seed should change the seed value."""

    gen = Normal2DGenerator()
    gen.set_seed(new_seed=7)

    assert gen.seed_value == 7


@pytest.mark.parametrize(
    "no_of_points", [0, 1, 2, 3]
)
def test_changed_seed_changed_value(no_of_points):
    """Setting a new seed should generate new values."""

    a = Normal2DGenerator().generate(no_of_points=no_of_points)
    gen = Normal2DGenerator()
    gen.set_seed(new_seed=127)
    b = gen.generate(no_of_points=no_of_points)

    assert not equal(a, b).any()


@pytest.mark.parametrize(
    "no_of_points", [0, 1, 2, 3]
)
def test_output_shape(no_of_points):
    """Tests the shape of the data generated."""

    a = Normal2DGenerator().generate(no_of_points=no_of_points)
    assert a.shape == (no_of_points, 2)


def test_randomized_init_seed():
    """Initializing as randomized should set seed value as None."""

    b = Normal2DGenerator(random=True)

    with pytest.raises(AttributeError):
        _ = b.seed_value


@pytest.mark.parametrize(
    "no_of_points", [0, 1, 2, 3]
)
def test_randomized_init_generation(no_of_points):
    """New instances must generate different first value to old instances if randomized."""

    a = Normal2DGenerator().generate(no_of_points=no_of_points)
    b = Normal2DGenerator(random=True).generate(no_of_points=no_of_points)

    assert not equal(a, b).any()


def test_same_random_state_gen_value():
    """New instances with same random state generates same values."""

    a = Normal2DGenerator(random_state=7).generate()
    b = Normal2DGenerator(random_state=7).generate()

    assert equal(a, b).all()


def test_diff_random_state_gen_value():
    """New instances with different random state generates different values."""

    a = Normal2DGenerator(random_state=7).generate()
    b = Normal2DGenerator(random_state=11).generate()

    assert not equal(a, b).any()
