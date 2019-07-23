import numpy as np
import matplotlib.pyplot as plt
from astropy import constants as const
from astropy import units as q

from ExoOrbit import Orbit

if __name__ == "__main__":
    a = q.AU.to("km")
    p = q.year.to("day")
    e = 0.1
    i = np.pi / 2
    w = np.pi / 2 + 0.1

    r_s = const.R_sun.to("km").value
    m_s = const.M_sun.to("kg").value
    t_s = 5770
    r_p = const.R_earth.to("km").value
    m_p = const.M_earth.to("kg").value

    orbit = Orbit(a, p, e, i, w, r_s, m_s, t_s, r_p, m_p)

    t = np.linspace(0, p, 1000)

    m = orbit.mean_anomaly(t)
    e = orbit.eccentric_anomaly(t)
    f = orbit.true_anomaly(t)
    d = orbit.distance(t)
    r = orbit.phase_angle(t)

    t0 = orbit.time_primary_transit()
    t0 = orbit.time_secondary_eclipse()
    t1 = orbit.first_contact()
    t4 = orbit.fourth_contact()
    print(t1)
    print(t4)

    plt.plot(t, r)

    # plt.hlines(r_s + r_p, 0, p, colors="r")
    plt.vlines((t1, t4), 0, np.max(r), colors="r")
    plt.vlines(t0, 0, np.max(r), colors="g")

    plt.show()

    pass
