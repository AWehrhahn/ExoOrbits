from typing import Tuple
from exoorbit.bodies import Planet, Star
import numpy as np
from scipy.optimize import minimize
from scipy.constants import G, c
from astropy import units as u
from astropy.time import Time

import numba as nb
from numba import njit, generated_jit


from .util import time_input

# based on http://sredfield.web.wesleyan.edu/jesse_tarnas_thesis_final.pdf
# and the exoplanet handbook

# TODO: Perturbations by other planets
# TODO: relativistic effects?

mjup_per_msun = (u.M_jup / u.M_sun).to(u.one)
day_per_year = (u.day / u.year).to(u.one)


@njit(nogil=True)
def _true_anomaly(t: np.ndarray, t0: float, p: float, e: float) -> np.ndarray:
    root = np.sqrt((1 + e) / (1 - e))
    ea = _eccentric_anomaly(t, t0, p, e)
    f = 2 * np.arctan(root * np.tan(ea / 2))
    return f


@njit(nogil=True)
def _mean_anomaly(t: np.ndarray, t0: float, p: float) -> np.ndarray:
    m = 2 * np.pi * (t - t0) / p
    m %= 2 * np.pi
    return m


@generated_jit(nopython=True, cache=True)
def _eccentric_anomaly(t: np.ndarray, t0: float, p: float, e: float) -> np.ndarray:
    if isinstance(t, nb.types.Float):
        return _eccentric_anomaly_scalar
    else:
        return _eccentric_anomaly_array

@njit(nogil=True)
def _eccentric_anomaly_scalar(t: float, t0: float, p: float, e: float) -> float:
    m = _mean_anomaly(t, t0, p)
    tolerance = 1e-8
    e = 0
    en = 10 * tolerance
    for _ in range(10):
        e = en
        en = m + e * np.sin(e)
        if np.abs(en - e) > tolerance:
            break
    en = ((en + np.pi) % (2 * np.pi)) - np.pi
    return en


@njit(nogil=True)
def _eccentric_anomaly_array(
    t: np.ndarray, t0: float, p: float, e: float
) -> np.ndarray:
    m = _mean_anomaly(t, t0, p)
    tolerance = 1e-8
    e = np.zeros_like(m)
    en = np.ones_like(m) * 10 * tolerance
    for _ in range(10):
        e = en
        en = m + e * np.sin(e)
        if np.any(np.abs(en - e) > tolerance):
            break

    en = ((en + np.pi) % (2 * np.pi)) - np.pi
    return en


@generated_jit(nopython=True)
def _phase_angle(
    t: np.ndarray, t0: float, p: float, e: float, i: float, w: float
) -> np.ndarray:
    if isinstance(t, nb.types.Float):
        return _phase_angle_scalar
    else:
        return _phase_angle_array


@njit(nogil=True)
def _phase_angle_array(
    t: np.ndarray, t0: float, p: float, e: float, i: float, w: float
) -> np.ndarray:
    # Determine whether the time is before or after transit
    k = (t - t0) % p
    k = k / p
    k = np.where(k < 0.5, 1, -1)
    # Calculate the angle
    f = _true_anomaly(t, t0, p, e)
    theta = np.arccos(np.sin(w + f) * np.sin(i))
    theta *= k
    return theta


@njit(nogil=True)
def _phase_angle_scalar(
    t: float, t0: float, p: float, e: float, i: float, w: float
) -> float:
    # Determine whether the time is before or after transit
    k = (t - t0) % p
    k = k / p
    k = 1 if k < 0.5 else -1
    # Calculate the angle
    f = _true_anomaly(t, t0, p, e)
    theta = np.arccos(np.sin(w + f) * np.sin(i))
    theta *= k
    return theta


@njit(nogil=True)
def _distance(t: np.ndarray, t0: float, a: float, p: float, e: float) -> np.ndarray:
    ea = _eccentric_anomaly(t, t0, p, e)
    d = a * (1 - e * np.cos(ea))
    return d


@njit(nogil=True)
def _projected_radius(
    t: np.ndarray, t0: float, a: float, p: float, e: float, i: float, w: float
) -> np.ndarray:
    theta = _phase_angle(t, t0, p, e, i, w)
    d = _distance(t, t0, a, p, e)
    r = np.abs(d * np.sin(theta))
    return r


@njit(nogil=True)
def _stellar_surface_covered_by_planet(
    t: np.ndarray,
    t0: float,
    a: float,
    rpl: float,
    rst: float,
    p: float,
    e: float,
    i: float,
    w: float,
) -> np.ndarray:
    d = _projected_radius(t, t0, a, p, e, i, w)
    area = np.zeros_like(t)

    # Use these to make it a bit more readable
    R = rst
    r = rpl
    # Case 1: planet completely inside the disk
    area[d + r <= R] = r ** 2 / R ** 2
    # Case 2: planet completely outside the disk
    area[d - r >= R] = 0
    # Case 3: inbetween
    select = (d + r > R) & (d - r < R)
    dp = d[select]
    area1 = r ** 2 * np.arccos((dp ** 2 + r ** 2 - R ** 2) / (2 * dp * r))
    area2 = R ** 2 * np.arccos((dp ** 2 + R ** 2 - r ** 2) / (2 * dp * R))
    area3 = 0.5 * np.sqrt((-dp + r + R) * (dp + r - R) * (dp - r + R) * (dp + r + R))
    area[select] = (area1 + area2 - area3) / (np.pi * R ** 2)
    return area


@njit(nogil=True)
def _radial_velocity_semiamplitude_planet(m_s, m_p, p, e, i):
    # m_s in solar masses
    # m_p in jupiter masses
    # p in days
    # return in m/s
    m = m_s / mjup_per_msun * (m_s + m_p * mjup_per_msun) ** (-2 / 3)
    b = np.sin(i) / np.sqrt(1 - e ** 2)
    t = (p * day_per_year) ** (-1 / 3)
    return 28.4329 * m * b * t


@njit(nogil=True)
def _radial_velocity_planet(t, t0, m_s, m_p, p, e, i, w):
    K = _radial_velocity_semiamplitude_planet(m_s, m_p, p, e, i)
    f = _true_anomaly(t, t0, p, e)
    rv = K * (np.cos(w + f) + e * np.cos(w))
    return rv


class Orbit:
    def __init__(self, star: Star, planet: Planet):
        """Calculates the orbit of an exoplanet

        Parameters
        ----------
        star : Star
            Central body (star) of the system
        planet : Planet
            Orbiting body (planet) of the system
        """
        self.star = star
        self.planet = planet
        self.star._orbit = self
        self.planet._orbit = self

        # TODO define these parameters
        self.v_s = 0
        self.albedo = 1

    @property
    def a(self) -> u.km:
        return self.planet.sma

    @property
    def p(self) -> u.day:
        return self.planet.period

    @property
    def e(self) -> u.one:
        return self.planet.ecc

    @property
    def i(self) -> u.deg:
        return self.planet.inc

    @property
    def w(self) -> u.deg:
        return self.planet.omega

    @property
    def t0(self) -> Time:
        return self.planet.t0

    @property
    def r_s(self) -> u.km:
        return self.star.radius

    @property
    def m_s(self) -> u.kg:
        return self.star.mass

    @property
    def r_p(self) -> u.km:
        return self.planet.radius

    @property
    def m_p(self) -> u.kg:
        return self.planet.mass

    @property
    def k(self) -> u.one:
        return self.r_p / self.r_s

    @time_input
    def z(self, t: Time) -> u.one:
        """projected radius in units of the stellar disk"""
        z = self.projected_radius(t) / self.r_s
        return z.decompose()

    @u.quantity_input
    def periapsis_distance(self) -> u.km:
        """Closest distance in the orbit"""
        return (1 - self.e) * self.a

    @u.quantity_input
    def apoapsis_distance(self) -> u.km:
        """Furthest distance in the orbit"""
        return (1 + self.e) * self.a

    @time_input
    def mean_anomaly(self, t: Time) -> u.rad:
        m = _mean_anomaly(t.mjd, self.t0.mjd, self.p.to_value(u.day))
        m = m << u.rad
        return m

    @time_input
    def true_anomaly(self, t: Time) -> u.rad:
        f = _true_anomaly(
            t.mjd, self.t0.mjd, self.p.to_value(u.day), self.e.to_value(u.one),
        )
        f = f << u.rad
        return f

    @time_input
    def eccentric_anomaly(self, t: Time) -> u.rad:
        en = _eccentric_anomaly(
            t.mjd, self.t0.mjd, self.p.to_value(u.day), self.e.to_value(u.one)
        )
        en = en << u.rad
        return en

    @time_input
    def distance(self, t: Time) -> u.km:
        """Distance from the center of the star to the center of the planet

        Parameters
        ----------
        t : float, array
            time in mjd

        Returns
        -------
        distance : float, array
            distance in km
        """
        d = _distance(
            t.mjd,
            self.t0.mjd,
            self.a.to_value(u.km),
            self.p.to_value(u.day),
            self.e.to_value(u.one),
        )
        d = d << u.km
        return d

    @time_input
    def phase_angle(self, t: Time) -> u.rad:
        """
        The phase angle describes the angle between
        the vector of observer’s line-of-sight and the
        vector from star to planet

        Parameters
        ----------
        t : float, array
            observation times in jd

        Returns
        -------
        phase_angle : float
            phase angle in radians
        """
        theta = _phase_angle(
            t.mjd,
            self.t0.mjd,
            self.p.to_value(u.day),
            self.e.to_value(u.one),
            self.i.to_value(u.rad),
            self.w.to_value(u.rad),
        )
        theta = theta << u.rad
        return theta

    @time_input
    def projected_radius(self, t: Time) -> u.km:
        """
        Distance from the center of the star to the center of the planet,
        i.e. distance projected on the stellar disk

        Parameters
        ----------
        t : float, array
            time in mjd

        Returns
        -------
        r : float, array
            distance in km
        """
        r = _projected_radius(
            t.mjd,
            self.t0.mjd,
            self.a.to_value(u.km),
            self.p.to_value(u.day),
            self.e.to_value(u.one),
            self.i.to_value(u.rad),
            self.w.to_value(u.rad),
        )
        r = r << u.km
        return r

    @time_input
    def position_3D(self, t: Time) -> Tuple[u.Quantity, u.Quantity, u.Quantity]:
        """Calculate the 3D position of the planet

        the coordinate system is centered in the star, x is towards the observer, z is "north", and y to the "right"

          z ^
            |
            | -¤-
            |̣_____>
            /      y
           / x

        Parameters:
        ----------
        t : float, array
            time in mjd

        Returns
        -------
        x, y, z: float, array
            position in stellar radii
        """
        # TODO this is missing the argument of periapsis
        phase = self.phase_angle(t)
        r = self.distance(t)
        i = self.i
        x = -r * np.cos(phase) * np.sin(i)
        y = -r * np.sin(phase)
        z = -r * np.cos(phase) * np.cos(i)
        return x, y, z

    @time_input
    def mu(self, t: Time) -> u.one:
        # mu = np.cos(self.phase_angle(t))
        r = self.projected_radius(t) / self.r_s
        r = r.decompose()
        mu = np.full_like(r, -1.0)
        mu[r <= 1] = np.sqrt(1 - r[r <= 1] ** 2)
        mu[r > 1] = -np.sqrt(-1 + r[r > 1] ** 2)
        return mu

    @time_input
    def stellar_surface_covered_by_planet(self, t: Time) -> u.one:
        area = _stellar_surface_covered_by_planet(
            t.mjd,
            self.t0.mjd,
            self.a.to_value(u.km),
            self.r_p.to_value(u.km),
            self.r_s.to_value(u.km),
            self.p.to_value(u.day),
            self.e.to_value(u.one),
            self.i.to_value(u.rad),
            self.w.to_value(u.rad),
        )
        area = area << u.one
        return area

    def _find_contact(self, r: u.km, bounds: Tuple[float, float]) -> Time:
        func = lambda t: np.abs(
            (self.projected_radius(Time(t, format="mjd")) - r).value
        )
        bounds = [bounds[0].mjd, bounds[1].mjd]
        t0 = bounds[0] + (bounds[1] - bounds[0]) / 4
        res = minimize(func, [t0], bounds=[bounds], method="Powell")
        res = Time(res.x[0], format="mjd")
        return res

    def first_contact(self) -> Time:
        """
        First contact is when the outer edge of the planet touches the stellar disk,
        i.e. when the transit curve begins

        Returns
        -------
        t1 : float
            time in mjd
        """
        t0 = self.time_primary_transit()
        r = self.r_s + self.r_p
        b = (t0 - self.p / 4, t0)
        return self._find_contact(r, b)

    def second_contact(self) -> Time:
        """
        Second contact is when the planet is completely in the stellar disk for the first time

        Returns
        -------
        t2 : float
            time in mjd
        """
        t0 = self.time_primary_transit()
        r = self.r_s - self.r_p
        b = (t0 - self.p / 4, t0)
        return self._find_contact(r, b)

    def third_contact(self) -> Time:
        """
        Third contact is when the planet begins to leave the stellar disk,
        but is still completely within the disk

        Returns
        -------
        t3 : float
            time in mjd
        """
        t0 = self.time_primary_transit()
        r = self.r_s - self.r_p
        b = (t0, t0 + self.p / 4)
        return self._find_contact(r, b)

    def fourth_contact(self) -> Time:
        """
        Fourth contact is when the planet completely left the stellar disk

        Returns
        -------
        t4 : float
            time in mjd
        """
        t0 = self.time_primary_transit()
        r = self.r_s + self.r_p
        b = (t0, t0 + self.p / 4)
        return self._find_contact(r, b)

    @time_input
    def transit_depth(self, t: Time) -> u.one:
        # r / r_s
        z = self.z(t)
        # r_p / r_s
        k = self.k

        depth = np.zeros(t.shape)

        # Planet fully inside the stellar disk
        depth[z <= (1 - k)] = k ** 2

        # Planet is entering or exiting the disk
        mask = (abs(1 - k) < z) & (z <= (1 + k))
        if np.any(mask):
            z = z[mask]
            k2, z2 = k ** 2, z ** 2
            kappa1 = np.arccos((1 - k2 + z2) / (2 * z)).to_value(u.rad)
            kappa0 = np.arccos((k2 + z2 - 1) / (2 * k * z)).to_value(u.rad)
            root = 0.5 * np.sqrt(4 * z2 - (1 - k2 + z2) ** 2)
            depth[mask] = 1 / np.pi * (k2 * kappa0 + kappa1 - root)

        return depth

    @u.quantity_input
    def impact_parameter(self) -> u.one:
        """
        The impact parameter is the shortest projected distance during a transit,
        i.e. how close the planet gets to the center of the star

        This will be 0 if the inclination is 90 deg

        Returns
        -------
        b : float
            distance in km
        """
        d = self.a / self.r_s * np.cos(self.i)
        e = (1 - self.e ** 2) / (1 + self.e * np.sin(self.w))
        return d * e

    @u.quantity_input
    def transit_time_total_circular(self) -> u.day:
        """
        The total time spent in transit for a circular orbit,
        i.e. if eccentricity where 0

        This should be the same as first contact to fourth contact
        There is only an analytical formula for the circular orbit, which is why this exists

        Returns
        -------
        t : float
            time in days
        """
        b = self.impact_parameter()
        alpha = self.r_s / self.a * np.sqrt((1 + self.k) ** 2 - b ** 2) / np.sin(self.i)
        return self.p / np.pi * np.arcsin(alpha)

    @u.quantity_input
    def transit_time_full_circular(self) -> u.day:
        """
        The total time spent in full transit for a circular orbit,
        i.e. the time during which the planet is completely inside the stellar disk
        if eccentricity where 0

        This should be the same as second contact to third contact
        There is only an analytical formula for the circular orbit, which is why this exists

        Returns
        -------
        t : float
            time in days
        """
        b = self.impact_parameter()
        alpha = self.r_s / self.a * np.sqrt((1 - self.k) ** 2 - b ** 2) / np.sin(self.i)
        return self.p / np.pi * np.arcsin(alpha)

    def time_primary_transit(self) -> Time:
        """
        The time of the primary transit,
        should be the same as t0

        Returns
        -------
        time : float
            time in mjd
        """
        b = (self.t0 - self.p / 4, self.t0 + self.p / 4)
        return self._find_contact(0, b)

    def time_secondary_eclipse(self) -> Time:
        return self.t0 + self.p / 2 * (1 + 4 * self.e * np.cos(self.w))

    @u.quantity_input
    def impact_parameter_secondary_eclipse(self) -> u.one:
        return (
            self.a
            * np.cos(self.i)
            / self.r_s
            * (1 - self.e ** 2)
            / (1 - self.e * np.sin(self.w))
        )

    @time_input
    def reflected_light_fraction(self, t: Time) -> u.one:
        return (
            self.albedo
            / 2
            * self.r_p ** 2
            / self.distance(t) ** 2
            * (1 + np.cos(self.phase_angle(t)))
        )

    def gravity_darkening_coefficient(self) -> u.one:
        t_s = self.star.teff
        return np.log10(G * self.m_s / self.r_s ** 2) / np.log10(t_s)

    @time_input
    def ellipsoid_variation_flux_fraction(self, t: Time) -> u.one:
        beta = self.gravity_darkening_coefficient()
        return (
            beta
            * self.m_p
            / self.m_s
            * (self.r_s / self.distance(t)) ** 3
            * (np.cos(self.w + self.true_anomaly(t)) * np.cos(self.i)) ** 2
        )

    @time_input
    def doppler_beaming_flux_fraction(self, t: Time) -> u.one:
        rv = self.radial_velocity_star(t)
        return 4 * rv / c

    @time_input
    def radial_velocity_planet(self, t: Time) -> u.km / u.s:
        """
        Radial velocity of the planet in the restframe of the host star
        Positive radial velocity means the planet is moving towards the observer,
        and negative values mean the planet is moving away

        Parameters
        ----------
        t : float, array
            times to evaluate in mjd

        Returns
        -------
        rv : float
            radial velocity in m/s
        """
        rv = _radial_velocity_planet(
            t.mjd,
            self.t0.mjd,
            self.m_s.to_value(u.M_sun),
            self.m_p.to_value(u.M_jup),
            self.p.to_value(u.day),
            self.e.to_value(u.one),
            self.i.to_value(u.rad),
            self.w.to_value(u.rad),
        )
        rv = rv << (u.m / u.s)
        return rv

    @time_input
    def radial_velocity_star(self, t: Time) -> u.km / u.s:
        """Radial velocity of the star

        Parameters
        ----------
        t : float, array
            times to evaluate in mjd

        Returns
        -------
        rv : float
            radial velocity in m/s
        """
        K = self.radial_velocity_semiamplitude()
        f = self.true_anomaly(t)
        rv = K * (np.cos(self.w + f) + self.e * np.cos(self.w))
        return rv

    @u.quantity_input
    def radial_velocity_semiamplitude(self) -> u.km / u.s:
        """Radial velocity semiamplitude of the star

        Returns
        -------
        K : float
            radial velocity semiamplitude in m/s
        """
        m = self.m_p / u.M_jup * ((self.m_s + self.m_p) / u.M_sun) ** (-2 / 3)
        b = np.sin(self.i) / np.sqrt(1 - self.e ** 2)
        t = (self.p / u.year) ** (-1 / 3)
        return 28.4329 * m * b * t * (u.m / u.s)

    @u.quantity_input
    def radial_velocity_semiamplitude_planet(self) -> u.km / u.s:
        """Radial velocity semiamplitude of the planet

        Returns
        -------
        K : float
            radial velocity semiamplitude in m/s
        """
        K = _radial_velocity_semiamplitude_planet(
            self.m_s.to_value(u.M_sun),
            self.m_p.to_value(u.M_jup),
            self.p.to_value(u.day),
            self.e.to_value(u.one),
            self.i.to_value(u.rad),
        )
        K = K << (u.m / u.s)
        return K
