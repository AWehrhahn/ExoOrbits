import pytest
import numpy as np

@pytest.fixture
def times(planet):
    t0 = planet.time_of_transit
    p = planet.period
    t = np.linspace(t0 - p / 2, t0 + p / 2, 1001)
    return t


@pytest.fixture
def t0(planet):
    return planet.time_of_transit


def test_mean_anomaly(orbit, times, t0):
    m = orbit.mean_anomaly(times)
    assert isinstance(m, np.ndarray)
    assert m.ndim == times.ndim
    assert m.size == times.size

    m = orbit.mean_anomaly(t0)
    assert isinstance(m, (float, np.floating))
    assert m == 0


def test_true_anomaly(orbit, times, t0):
    f = orbit.true_anomaly(times)
    assert isinstance(f, np.ndarray)
    assert f.ndim == times.ndim
    assert f.size == times.size
    assert np.all((f <= np.pi) & (f >= -np.pi))

    f = orbit.true_anomaly(t0)
    assert isinstance(f, (float, np.floating))
    # assert f == 0 # Close to 0?


def test_eccentric_anomaly(orbit, times, t0):
    ea = orbit.eccentric_anomaly(times)
    assert isinstance(ea, np.ndarray)
    assert ea.ndim == times.ndim
    assert ea.size == times.size
    assert np.all((ea <= np.pi) & (ea >= -np.pi))

    ea = orbit.eccentric_anomaly(t0)
    assert isinstance(ea, (float, np.floating))
    assert np.isclose(ea, 0)


def test_distance(orbit, times, planet):
    d = orbit.distance(times)
    assert isinstance(d, np.ndarray)
    assert d.ndim == times.ndim
    assert d.size == times.size

    min_distance = orbit.periapsis_distance()
    max_distance = orbit.apoapsis_distance()
    assert np.all((d <= max_distance) & (d >= min_distance))


def test_phase_angle(orbit, times, t0):
    pa = orbit.phase_angle(times)
    assert isinstance(pa, np.ndarray)
    assert pa.ndim == times.ndim
    assert pa.size == times.size
    assert np.all((pa >= -np.pi) & (pa <= np.pi))

    pa = orbit.phase_angle(t0)
    assert isinstance(pa, (float, np.floating))
    assert pa == 0


def test_radius(orbit, times, t0):
    r = orbit.projected_radius(times)
    max_d = orbit.apoapsis_distance()

    assert isinstance(r, np.ndarray)
    assert r.ndim == times.ndim
    assert r.size == times.size
    assert np.all((r >= 0) & (r <= max_d))

    r = orbit.phase_angle(t0)
    b = orbit.impact_parameter()
    assert isinstance(r, (float, np.floating))
    assert np.isclose(r, b)


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
    assert np.all((mu == -1) | ((mu >= 0) & (mu <= 1)))


def test_contact(orbit):
    t0 = orbit.time_primary_transit()
    t1 = orbit.first_contact()
    t2 = orbit.second_contact()
    t3 = orbit.third_contact()
    t4 = orbit.fourth_contact()

    assert isinstance(t0, (float, np.floating))
    assert isinstance(t1, (float, np.floating))
    assert isinstance(t2, (float, np.floating))
    assert isinstance(t3, (float, np.floating))
    assert isinstance(t4, (float, np.floating))
    # Assure that the order is correct
    assert t1 < t2 < t0 < t3 < t4


def test_impact_parameter(orbit, star):
    b = orbit.impact_parameter()
    r_s = star.radius
    assert isinstance(b, (float, np.floating))
    assert 0 <= b <= r_s
