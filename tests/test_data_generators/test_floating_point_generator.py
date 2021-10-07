from vizml.data_generator import FloatingPointGenerator


def test_same_intial_generation():
    """New instances must generate same first value as old instances."""

    a = FloatingPointGenerator().generate()
    b = FloatingPointGenerator().generate()

    assert a == b


def test_setting_seed():
    """Setting seed should change the seed value."""

    gen = FloatingPointGenerator()
    gen.set_seed(new_seed=7)

    assert gen.seed_value == 7


def test_changed_seed_changed_value():
    """Setting a new seed should generate new value."""

    a = FloatingPointGenerator().generate()
    gen = FloatingPointGenerator()
    gen.set_seed(new_seed=127)
    b = gen.generate()

    assert a != b


def test_randomized_init_seed():
    """Initializing as randomized should set a random seed."""

    a = FloatingPointGenerator()
    b = FloatingPointGenerator(random=True)

    assert a.seed_value != b.seed_value


def test_randomized_init_generation():
    """New instances must generate different first value to old instances if randomized."""

    a = FloatingPointGenerator().generate()
    b = FloatingPointGenerator(random=True).generate()

    assert a != b
