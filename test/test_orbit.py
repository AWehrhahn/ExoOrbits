import pytest
import numpy as np

from astropy.time import Time
from astropy import units as u


@pytest.fixture
def times(planet):
    t0 = planet.time_of_transit.mjd
    p = planet.period.to_value("day")
    t = np.linspace(t0 - p / 2, t0 + p / 2, 1001)
    t = Time(t, format="mjd")
    return t


@pytest.fixture
def t0(planet):
    return planet.time_of_transit


def test_mean_anomaly(orbit, times, t0):
    m = orbit.mean_anomaly(times)
    assert isinstance(m, u.quantity.Quantity)
    assert m.to("rad")
    assert m.ndim == times.ndim
    assert m.size == times.size

    m = orbit.mean_anomaly(t0)
    assert isinstance(m, u.quantity.Quantity)
    assert m.to("rad")
    assert m == 0 * u.rad


def test_true_anomaly(orbit, times, t0):
    f = orbit.true_anomaly(times)
    assert isinstance(f, u.quantity.Quantity)
    assert f.to("rad")
    assert f.ndim == times.ndim
    assert f.size == times.size
    assert np.all((f <= np.pi * u.rad) & (f >= -np.pi * u.rad))

    f = orbit.true_anomaly(t0)
    assert isinstance(f, u.quantity.Quantity)
    assert f.to("rad")
    # assert f == 0 # Close to 0?


def test_eccentric_anomaly(orbit, times, t0):
    ea = orbit.eccentric_anomaly(times)
    assert isinstance(ea, u.quantity.Quantity)
    assert ea.to("rad")
    assert ea.ndim == times.ndim
    assert ea.size == times.size
    assert np.all((ea <= np.pi * u.rad) & (ea >= -np.pi * u.rad))

    ea = orbit.eccentric_anomaly(t0)
    assert isinstance(ea, u.quantity.Quantity)
    assert ea.to("rad")
    assert np.isclose(ea.to_value("rad"), 0)


def test_distance(orbit, times, planet):
    d = orbit.distance(times)
    assert isinstance(d, u.quantity.Quantity)
    assert d.to("m")
    assert d.ndim == times.ndim
    assert d.size == times.size

    min_distance = orbit.periapsis_distance()
    max_distance = orbit.apoapsis_distance()
    assert np.all((d <= max_distance) & (d >= min_distance))


def test_phase_angle(orbit, times, t0):
    pa = orbit.phase_angle(times)
    assert isinstance(pa, u.quantity.Quantity)
    assert pa.to("rad")
    assert pa.ndim == times.ndim
    assert pa.size == times.size
    assert np.all((pa >= -np.pi * u.rad) & (pa <= np.pi * u.rad))

    pa = orbit.phase_angle(t0)
    assert isinstance(pa, u.quantity.Quantity)
    assert pa.to("rad")
    assert pa.to_value("rad") == 0


def test_radius(orbit, times, t0):
    r = orbit.projected_radius(times)
    max_d = orbit.apoapsis_distance()

    assert isinstance(r, u.quantity.Quantity)
    assert r.to(u.m)
    assert r.ndim == times.ndim
    assert r.size == times.size
    assert np.all((r >= 0 * u.m) & (r <= max_d))

    r = orbit.phase_angle(t0)
    b = orbit.impact_parameter()
    assert isinstance(r, u.quantity.Quantity)
    assert r.to("deg")
    assert np.isclose(r.to_value("rad"), b.to_value(1))


def test_position3d(orbit, times):
    # x is towards the observer, z is "north", and y to the "right"
    x, y, z = orbit.position_3D(times)
    d = orbit.distance(times)
    max_d = orbit.apoapsis_distance()

    assert isinstance(x, np.ndarray)
    assert x.ndim == times.ndim
    assert x.size == times.size
    assert np.all(np.abs(x) <= max_d)
    assert isinstance(y, np.ndarray)
    assert y.ndim == times.ndim
    assert y.size == times.size
    assert np.all(np.abs(y) <= max_d)
    assert isinstance(z, np.ndarray)
    assert z.ndim == times.ndim
    assert z.size == times.size
    assert np.all(np.abs(z) <= max_d)

    # Results are self consistent (within numerical uncertainties)
    assert np.allclose(x ** 2 + y ** 2 + z ** 2, d ** 2)


def test_mu(orbit, times):
    mu = orbit.mu(times)

    assert isinstance(mu, np.ndarray)
    assert mu.ndim == times.ndim
    assert mu.size == times.size
    assert np.all(mu <= 1)


def test_contact(orbit):
    t0 = orbit.time_primary_transit()
    t1 = orbit.first_contact()
    t2 = orbit.second_contact()
    t3 = orbit.third_contact()
    t4 = orbit.fourth_contact()

    assert isinstance(t0, Time)
    assert isinstance(t1, Time)
    assert isinstance(t2, Time)
    assert isinstance(t3, Time)
    assert isinstance(t4, Time)
    # Assure that the order is correct
    assert t1 < t2 < t0 < t3 < t4


def test_impact_parameter(orbit, star):
    b = orbit.impact_parameter()
    r_s = star.radius
    assert isinstance(b, u.quantity.Quantity)
    assert b.to(1)
    assert 0 <= b <= 1
