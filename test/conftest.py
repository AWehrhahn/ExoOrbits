import pytest

from exoorbit import Orbit
from exoorbit.library import Sun, Earth, GJ1214, GJ1214_b


@pytest.fixture(params=[(Sun, Earth), (GJ1214, GJ1214_b)], ids=["Sun_Earth", "GJ1214"])
def system(request):
    return request.param


@pytest.fixture
def star(system):
    return system[0]


@pytest.fixture
def planet(system):
    return system[1]


@pytest.fixture
def orbit(star, planet):
    return Orbit(star, planet)
