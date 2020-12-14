import numpy as np
from scipy.optimize import fsolve, minimize_scalar
from scipy.constants import G, c
from astropy import constants as const
from astropy import units as u
from astropy.time import Time

from .util import cache, hasCache, time_input

pi = np.pi

# based on http://sredfield.web.wesleyan.edu/jesse_tarnas_thesis_final.pdf
# and the exoplanet handbook


# TODO: Perturbations by other planets
# TODO: relativistic effects?
# TODO: Determine the speed of the cache, is it too slow?
# TODO: Empty cache when self.parameters change


class Orbit(hasCache):
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
        self.star._orbit = self
        self.planet._orbit = self

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

    @time_input
    def z(self, t):
        return self.projected_radius(t) / self.r_s

    def periapsis_distance(self):
        """Closest distance in the orbit"""
        return (1 - self.e) * self.a

    def apoapsis_distance(self):
        """Furthest distance in the orbit"""
        return (1 + self.e) * self.a

    @time_input
    def mean_anomaly(self, t):
        m = 2 * pi * (t - self.t0) / self.p
        m = m * u.rad
        return m

    @time_input
    def true_anomaly(self, t):
        root = np.sqrt((1 + self.e) / (1 - self.e))
        ea = self.eccentric_anomaly(t)
        f = 2 * np.arctan(root * np.tan(ea / 2))
        return f

    @time_input
    def eccentric_anomaly(self, t):
        m = self.mean_anomaly(t)

        tolerance = 1e-8 * m.unit
        e = 0 * m.unit
        en = 10 * tolerance
        while np.any(np.abs(en - e) > tolerance):
            e = en
            en = m + self.e * np.sin(e) * m.unit

        en = ((en + np.pi * u.rad) % (2 * np.pi * u.rad)) - np.pi * u.rad
        return en

    @time_input
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

    @time_input
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
        # Determine whether the time is before or after transit
        p = self.p.to_value(u.day)
        k = (t - self.t0).jd % p
        k = k / p
        k = np.where(k < 0.5, 1, -1)
        # Calculate the angle
        f = self.true_anomaly(t)
        theta = np.arccos(np.sin(self.w + f) * np.sin(self.i))
        theta *= k
        return theta

    @time_input
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

    @time_input
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

    @time_input
    def mu(self, t):
        # mu = np.cos(self.phase_angle(t))
        r = self.projected_radius(t) / self.r_s
        r = r.decompose()
        mu = np.full_like(r, -1.0)
        mu[r <= 1] = np.sqrt(1 - r[r <= 1] ** 2)
        mu[r > 1] = -np.sqrt(-1 + r[r > 1] ** 2)
        return mu

    @time_input
    def stellar_surface_covered_by_planet(self, t):
        d = self.projected_radius(t)
        area = np.zeros(len(t)) << u.one

        # Use these to make it a bit more readable
        R = self.star.radius
        r = self.planet.radius
        # Case 1: planet completely inside the disk
        area[d + r <= R] = r ** 2 / R ** 2
        # Case 2: planet completely outside the disk
        area[d - r >= R] = 0
        # Case 3: inbetween
        select = (d + r > R) & (d - r < R)
        dp = d[select]
        area1 = r ** 2 * np.arccos((dp ** 2 + r ** 2 - R ** 2) / (2 * dp * r))
        area2 = R ** 2 * np.arccos((dp ** 2 + R ** 2 - r ** 2) / (2 * dp * R))
        area3 = (
            0.5
            * np.sqrt((-dp + r + R) * (dp + r - R) * (dp - r + R) * (dp + r + R))
            * u.rad
        )
        area[select] = (area1 + area2 - area3) / (np.pi * u.rad * R ** 2)

        return area

    def _find_contact(self, r, bounds):
        func = lambda t: abs(self.projected_radius(Time(t, format="mjd")) - r).value
        bounds = [bounds[0].mjd, bounds[1].mjd]
        res = minimize_scalar(
            func, bounds=bounds, method="bounded", options={"xatol": 1e-12}
        )
        res = Time(res.x, format="mjd")
        return res

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

    @time_input
    def transit_depth(self, t):
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
            kappa1 = np.arccos((1 - k2 + z2) / (2 * z))
            kappa0 = np.arccos((k2 + z2 - 1) / (2 * k * z))
            root = 0.5 * np.sqrt(4 * z2 - (1 - k2 + z2) ** 2)
            depth[mask] = 1 / pi * (k2 * kappa0 + kappa1 - root)

        return depth

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

    def time_secondary_eclipse(self):
        return self.p / 2 * (1 + 4 * self.e * np.cos(self.w))

    def impact_parameter_secondary_eclipse(self):
        return (
            self.a
            * np.cos(self.i)
            / self.r_s
            * (1 - self.e ** 2)
            / (1 - self.e * np.sin(self.w))
        )

    def reflected_light_fraction(self, t):
        return (
            self.albedo
            / 2
            * self.r_p ** 2
            / self.distance(t) ** 2
            * (1 + np.cos(self.phase_angle(t)))
        )

    def gravity_darkening_coefficient(self):
        t_s = self.star.teff
        return np.log10(G * self.m_s / self.r_s ** 2) / np.log10(t_s)

    @time_input
    def ellipsoid_variation_flux_fraction(self, t):
        beta = self.gravity_darkening_coefficient()
        return (
            beta
            * self.m_p
            / self.m_s
            * (self.r_s / self.distance(t)) ** 3
            * (np.cos(self.w + self.true_anomaly(t)) * np.cos(self.i)) ** 2
        )

    @time_input
    def doppler_beaming_flux_fraction(self, t):
        rv = self.radial_velocity_star(t)
        return 4 * rv / c

    @time_input
    def radial_velocity_planet(self, t):
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
        K = self.radial_velocity_semiamplitude_planet()
        f = self.true_anomaly(t)
        rv = K * (np.cos(self.w + f) + self.e * np.cos(self.w))
        return rv

    @time_input
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
        rv = K * (np.cos(self.w + f) + self.e * np.cos(self.w))
        return rv

    def radial_velocity_semiamplitude(self):
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

    def radial_velocity_semiamplitude_planet(self):
        """Radial velocity semiamplitude of the planet

        Returns
        -------
        K : float
            radial velocity semiamplitude in m/s
        """
        m = self.m_s / u.M_jup * ((self.m_s + self.m_p) / u.M_sun) ** (-2 / 3)
        b = np.sin(self.i) / np.sqrt(1 - self.e ** 2)
        t = (self.p / u.year) ** (-1 / 3)
        return 28.4329 * m * b * t * (u.m / u.s)
