from astropy import constants as c
from astropy import units as u
from astropy.time import Time

from .bodies import Planet, Star

# Section 1: Solar System bodies
# NOTE: inclination is set to 0 for Earth so that we get transits
Sun = Star(mass=c.M_sun, radius=c.R_sun, name="Sun", teff=5770 * u.K, useDatabase=False)
Earth = Planet(
    mass=c.M_earth,
    radius=c.R_earth,
    sma=1 * u.AU,
    period=1 * u.year,
    ecc=0.016 * u.one,
    name="Earth",
    useDatabase=False,
)
# Earth on a perfectly circular orbit
Earth_circular = Planet(
    mass=c.M_earth,
    radius=c.R_earth,
    sma=1 * u.AU,
    period=1 * u.year,
    name="Earth",
    useDatabase=False,
)

# Section 2: Exoplanet systems
GJ1214 = Star(
    mass=0.157 * c.M_sun,
    radius=0.2110 * c.R_sun,
    name="GJ1214",
    teff=3030 * u.K,
    useDatabase=False,
)
GJ1214_b = Planet(
    mass=0.0204 * c.M_jup,
    radius=0.239 * c.R_jup,
    sma=0.01433 * u.AU,
    period=1.58 * u.day,
    time_of_transit=Time(54980.248796, format="mjd"),
    name="GJ1214 b",
    useDatabase=False,
)
