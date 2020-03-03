from astropy import units as u
from astropy import constants as c

from .bodies import Body, Star, Planet

# Section 1: Solar System bodies
# NOTE: inclination is set to 0 for Earth so that we get transits
Sun = Star(c.M_sun, c.R_sun, "Sun", teff=5770)
Earth = Planet(
    c.M_earth,
    c.R_earth,
    1 * u.AU,
    1 * u.year,
    0.016,
    90 * u.deg,
    90 * u.deg,
    0,
    name="Earth",
)
# Earth on a perfectly circular orbit
Earth_circular = Planet(c.M_earth, c.R_earth, 1 * u.AU, 1 * u.year, name="Earth")

# Section 2: Exoplanet systems
GJ1214 = Star(0.157 * c.M_sun, 0.2110 * c.R_sun, "GJ1214", teff=3030)
GJ1214_b = Planet(
    0.0204 * c.M_jup,
    0.239 * c.R_jup,
    0.01433 * u.AU,
    1.58 * u.day,
    0,
    90 * u.deg,
    90 * u.deg,
    54980.248796,
    name="GJ1214 b",
)
