from functools import wraps, lru_cache

import numpy as np
from scipy.optimize import fsolve, minimize_scalar
from scipy.constants import G, c
from astropy import constants as const
from astropy import units as u

pi = np.pi
m_jup = const.M_jup.to("kg").value
m_sol = const.M_sun.to("kg").value

# based on http://sredfield.web.wesleyan.edu/jesse_tarnas_thesis_final.pdf
# and the exoplanet handbook


# TODO: Perturbations by other planets
# TODO: relativistic effects?
# TODO: Cache intermediate results


def cache(function):
    # Switch order of array and self, because only the first argument is used for the cache
    @lru_cache()
    def cached_wrapper(hashable_array, self):
        if hashable_array is not None:
            array = np.array(hashable_array)
            return function(self, array)
        else:
            return function(self)

    @wraps(function)
    def wrapper(self, array=None):
        if isinstance(array, np.ndarray):
            if array.ndim > 0:
                return cached_wrapper(tuple(array), self)
            else:
                return cached_wrapper(float(array), self)
        else:
            return cached_wrapper(array, self)

    # copy lru_cache attributes over too
    wrapper.cache_info = cached_wrapper.cache_info
    wrapper.cache_clear = cached_wrapper.cache_clear

    return wrapper


class Orbit:
    def __init__(self, star, planet):
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

        # TODO define these parameters
        self.v_s = 0
        self.albedo = 1

    @property
    def a(self):
        return self.planet.semi_major_axis

    @property
    def p(self):
        return self.planet.period

    @property
    def e(self):
        return self.planet.eccentricity

    @property
    def i(self):
        return self.planet.inclination

    @property
    def w(self):
        return self.planet.argument_of_periastron

    @property
    def t0(self):
        return self.planet.time_of_transit

    @property
    def r_s(self):
        return self.star.radius

    @property
    def m_s(self):
        return self.star.mass

    @property
    def r_p(self):
        return self.planet.radius

    @property
    def m_p(self):
        return self.planet.mass

    @property
    def k(self):
        return self.r_p / self.r_s

    @cache
    def z(self, t):
        return self.projected_radius(t) / self.r_s

    @cache
    def periapsis_distance(self):
        """Closest distance in the orbit"""
        return (1 - self.e) * self.a

    @cache
    def apoapsis_distance(self):
        """Furthest distance in the orbit"""
        return (1 + self.e) * self.a

    @cache
    def mean_anomaly(self, t):
        m = 2 * pi * (t - self.t0) / self.p
        return m

    @cache
    def true_anomaly(self, t):
        root = np.sqrt((1 + self.e) / (1 - self.e))
        ea = self.eccentric_anomaly(t)
        f = 2 * np.arctan(root * np.tan(ea / 2))
        return f

    @cache
    def eccentric_anomaly(self, t):
        # TODO cache results
        tolerance = 1e-8
        m = self.mean_anomaly(t)

        e = 0
        en = 10 * tolerance
        while np.any(np.abs(en - e) > tolerance):
            e = en
            en = m + self.e * np.sin(e)

        en = ((en + np.pi) % (2 * np.pi)) - np.pi
        # en = np.clip(en, -np.pi, np.pi)
        return en

    @cache
    def distance(self, t):
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
        return self.a * (1 - self.e * np.cos(self.eccentric_anomaly(t)))

    @cache
    def phase_angle(self, t):
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
        k = ((t - self.t0) % self.p) / self.p
        k = np.where(k < 0.5, 1, -1)
        f = self.true_anomaly(t)
        theta = np.arccos(np.sin(self.w + f) * np.sin(self.i))
        return k * theta

    @cache
    def projected_radius(self, t):
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
        theta = self.phase_angle(t)
        d = self.distance(t)
        r = np.abs(d * np.sin(theta))
        return r

    @cache
    def position_3D(self, t):
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

    @cache
    def mu(self, t):
        # mu = np.cos(self.phase_angle(t))
        r = self.projected_radius(t) / self.r_s
        mu = np.full_like(r, -1.)
        np.sqrt(1 - r ** 2, where=r <= 1, out=mu)
        return mu

    def _find_contact(self, r, bounds):
        func = lambda t: abs(self.projected_radius(t) - r)
        res = minimize_scalar(
            func, bounds=bounds, method="bounded", options={"xatol": 1e-12}
        )
        res = res.x
        return res

    @cache
    def first_contact(self):
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
        b = (t0 - self.p / 4, t0 - 1e-8)
        return self._find_contact(r, b)

    @cache
    def second_contact(self):
        """
        Second contact is when the planet is completely in the stellar disk for the first time

        Returns
        -------
        t2 : float
            time in mjd
        """
        t0 = self.time_primary_transit()
        r = self.r_s - self.r_p
        b = (t0 - self.p / 4, t0 - 1e-8)
        return self._find_contact(r, b)

    @cache
    def third_contact(self):
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
        b = (t0 + 1e-8, t0 + self.p / 4)
        return self._find_contact(r, b)

    @cache
    def fourth_contact(self):
        """
        Fourth contact is when the planet completely left the stellar disk

        Returns
        -------
        t4 : float
            time in mjd
        """
        t0 = self.time_primary_transit()
        r = self.r_s + self.r_p
        b = (t0 + 1e-8, t0 + self.p / 4)
        return self._find_contact(r, b)

    @cache
    def transit_depth(self, t):
        z = self.z(t)
        k = self.k

        depth = np.full_like(t, 1)
        depth[(1 + k) > z] = 0
        depth[z <= (1 - k)] = k ** 2

        mask = (abs(1 - k) < z) & (z <= (1 + k))
        if np.any(mask):
            z = z[mask]
            kappa1 = np.arccos((1 - k ** 2 + z ** 2) / (2 * z))
            kappa0 = np.arccos((k ** 2 + z ** 2 - 1) / (2 * k * z))
            root = np.sqrt((4 * z ** 2 - (1 + z ** 2 - k ** 2) ** 2) / 4)
            depth[mask] = 1 / pi * (k ** 2 * kappa0 + kappa1 - root)

        return depth

    @cache
    def impact_parameter(self):
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

    @cache
    def transit_time_total_circular(self):
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
        return self.p / pi * np.arcsin(alpha)

    @cache
    def transit_time_full_circular(self):
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
        return self.p / pi * np.arcsin(alpha)

    @cache
    def time_primary_transit(self):
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

    @cache
    def time_secondary_eclipse(self):
        return self.p / 2 * (1 + 4 * self.e * np.cos(self.w))

    @cache
    def impact_parameter_secondary_eclipse(self):
        return (
            self.a
            * np.cos(self.i)
            / self.r_s
            * (1 - self.e ** 2)
            / (1 - self.e * np.sin(self.w))
        )

    @cache
    def reflected_light_fraction(self, t):
        return (
            self.albedo
            / 2
            * self.r_p ** 2
            / self.distance(t) ** 2
            * (1 + np.cos(self.phase_angle(t)))
        )

    @cache
    def gravity_darkening_coefficient(self):
        t_s = self.star.teff
        return np.log10(G * self.m_s / self.r_s ** 2) / np.log10(t_s)

    @cache
    def ellipsoid_variation_flux_fraction(self, t):
        beta = self.gravity_darkening_coefficient()
        return (
            beta
            * self.m_p
            / self.m_s
            * (self.r_s / self.distance(t)) ** 3
            * (np.cos(self.w + self.true_anomaly(t)) * np.cos(self.i)) ** 2
        )

    @cache
    def doppler_beaming_flux_fraction(self, t):
        rv = self.radial_velocity_star(t)
        return 4 * rv / c

    @cache
    def radial_velocity_planet(self, t):
        """Radial velocity of the planet

        Parameters
        ----------
        t : float, array
            times to evaluate in mjd

        Returns
        -------
        rv : float
            radial velocity in m/s
        """
        K = self.radial_velocity_semiamplitude_planet()
        f = self.true_anomaly(t)
        return self.v_s + K * (np.cos(self.w + f) + self.e * np.cos(self.w))

    @cache
    def radial_velocity_star(self, t):
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
        return self.v_s + K * (np.cos(self.w + f) + self.e * np.cos(self.w))

    @cache
    def radial_velocity_semiamplitude(self):
        """Radial velocity semiamplitude of the star

        Returns
        -------
        K : float
            radial velocity semiamplitude in m/s
        """
        m = self.m_p / m_jup * ((self.m_s + self.m_p) / m_sol) ** (-2 / 3)
        b = np.sin(self.i) / np.sqrt(1 - self.e ** 2)
        t = (self.p / u.year.to(u.day)) ** (-1 / 3)
        return 28.4329 * m * b * t

    @cache
    def radial_velocity_semiamplitude_planet(self):
        """Radial velocity semiamplitude of the planet

        Returns
        -------
        K : float
            radial velocity semiamplitude in m/s
        """
        m = self.m_s / m_jup * ((self.m_s + self.m_p) / m_sol) ** (-2 / 3)
        b = np.sin(self.i) / np.sqrt(1 - self.e ** 2)
        t = (self.p / u.year.to(u.day)) ** (-1 / 3)
        return 28.4329 * m * b * t
