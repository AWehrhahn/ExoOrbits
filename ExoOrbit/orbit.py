import numpy as np
from scipy.optimize import fsolve, minimize_scalar
from scipy.constants import G, c
from astropy import constants as const

pi = np.pi
m_jup = const.M_jup.to("kg").value
m_sol = const.M_sun.to("kg").value

# based on http://sredfield.web.wesleyan.edu/jesse_tarnas_thesis_final.pdf
# and the exoplanet handbook


# TODO: Perturbations by other planets
# TODO: relativistic effects?


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
    def e(self):
        return self.planet.eccentricity

    @property
    def k(self):
        return self.r_p / self.r_s

    def z(self, t):
        return self.projected_radius(t) / self.r_s

    def mean_anomaly(self, t):
        m = self.t0 + 2 * pi * t / self.p
        return m

    def true_anomaly(self, t):
        e = self.e
        f = 2 * np.arctan(
            np.sqrt((1 + e) / (1 - e)) * np.tan(self.eccentric_anomaly(t) / 2)
        )
        return f

    def eccentric_anomaly(self, t):
        # TODO cache results
        tolerance = 1e-8
        m = self.mean_anomaly(t)

        e = 0
        en = 10 * tolerance
        while np.any(np.abs(en - e) > tolerance):
            e = en
            en = m + self.e * e
        return en

    def distance(self, t):
        return self.a * (1 - self.e * np.cos(self.eccentric_anomaly(t)))

    def phase_angle(self, t):
        """The phase angle describes the angle between
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
        return np.arccos(np.sin(self.w + self.true_anomaly(t)) * np.sin(self.i))

    def projected_radius(self, t):
        theta = self.phase_angle(t)
        d = self.distance(t)
        return d * np.sin(theta)

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
        phase : {float, np.ndarray}
            phase in radians
        Returns
        -------
        x, y, z: {float, np.ndarray}
            position in stellar radii
        """

        #TODO avoid duplicate calculation
        phase = self.phase_angle(t)
        r = self.projected_radius(t)
        i = self.i
        x = -r * np.cos(phase) * np.sin(i)
        y = -r * np.sin(phase)
        z = -r * np.cos(phase) * np.cos(i)
        return x, y, z

    def mu(self, t):
        r = self.projected_radius(t) / self.r_s
        tmp = 1 - r ** 2
        mu = np.full_like(r, -1)
        np.sqrt(tmp, where=tmp >= 0, out=mu)
        return mu

    def _find_contact(self, r, bounds):
        func = lambda t: abs(self.projected_radius(t) - r)
        res = minimize_scalar(
            func, bounds=bounds, method="bounded", options={"xatol": 1e-12}
        )
        res = res.x
        return res

    def first_contact(self):
        t0 = self.time_primary_transit()
        r = self.r_s + self.r_p
        b = (t0 - self.p / 4, t0 - 1e-8)
        return self._find_contact(r, b)

    def second_contact(self):
        t0 = self.time_primary_transit()
        r = self.r_s - self.r_p
        b = (t0 - self.p / 4, t0 - 1e-8)
        return self._find_contact(r, b)

    def third_contact(self):
        t0 = self.time_primary_transit()
        r = self.r_s - self.r_p
        b = (t0 + 1e-8, t0 + self.p / 4)
        return self._find_contact(r, b)

    def fourth_contact(self):
        t0 = self.time_primary_transit()
        r = self.r_s + self.r_p
        b = (t0 + 1e-8, t0 + self.p / 4)
        return self._find_contact(r, b)

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

    def impact_parameter(self):
        return (
            self.a
            * np.cos(self.i)
            / self.r_s
            * (1 - self.e ** 2)
            / (1 + self.e * np.sin(self.w))
        )

    def transit_time_total_circular(self):
        b = self.impact_parameter()
        return (
            self.p
            / pi
            * np.arcsin(
                self.r_s / self.a * np.sqrt((1 + self.k) ** 2 - b ** 2) / np.sin(self.i)
            )
        )

    def transit_time_full_circular(self):
        b = self.impact_parameter()
        return (
            self.p
            / pi
            * np.arcsin(
                self.r_s / self.a * np.sqrt((1 - self.k) ** 2 - b ** 2) / np.sin(self.i)
            )
        )

    def time_primary_transit(self):
        b = (-self.p/2, self.p/2)
        return self._find_contact(0, b)
        t0 = self.p * (1 + 4 * self.e * np.cos(self.w))
        return t0 % self.p

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

    def ellipsoid_variation_flux_fraction(self, t):
        beta = self.gravity_darkening_coefficient()
        return (
            beta
            * self.m_p
            / self.m_s
            * (self.r_s / self.distance(t)) ** 3
            * (np.cos(self.w + self.true_anomaly(t)) * np.cos(self.i)) ** 2
        )

    def doppler_beaming_flux_fraction(self, t):
        rv = self.radial_velocity_star(t)
        return 4 * rv / c

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

    def radial_velocity_semiamplitude(self):
        """Radial velocity semiamplitude of the star

        Returns
        -------
        K : float
            radial velocity semiamplitude in m/s
        """
        return (
            28.4329
            / np.sqrt(1 - self.e ** 2)
            * self.m_p
            * np.sin(self.i)
            / m_jup
            * ((self.m_s + self.m_p) / m_sol) ** (-2 / 3)
            * (self.p / 365.25) ** (-1 / 3)
        )

    def radial_velocity_semiamplitude_planet(self):
        """Radial velocity semiamplitude of the planet

        Returns
        -------
        K : float
            radial velocity semiamplitude in m/s
        """
        return (
            28.4329
            / np.sqrt(1 - self.e ** 2)
            * self.m_s
            * np.sin(self.i)
            / m_jup
            * ((self.m_s + self.m_p) / m_sol) ** (-2 / 3)
            * (self.p / 365.25) ** (-1 / 3)
        )

