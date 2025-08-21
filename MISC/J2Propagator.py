import numpy as np
from datetime import datetime, timedelta

def J2_Propagator(sat, dt_analysis_period, step_size):
    """
    J2-perturbed orbit propagator (ECI position/velocity at each epoch)
    Inputs:
        sat: dictionary containing satellite state (sma, ecc, inc, raan, argp, nu)
        dt_epoch: epoch datetime (datetime object)
        dt_analysis_period: (start_datetime, end_datetime)
        step_size: seconds
    Returns:
        ephemeris: [N x 7] array (dt, x, y, z, vx, vy, vz)
        orbit_period_sec: orbital period [sec]
        deg_inc: inclination [deg]
        deg_raan: RAAN [deg]
        deg_w: argument of perigee [deg]
        deg_TA: true anomaly [deg]
        dt_epoch: epoch datetime (datetime object)
        dt_analysis_period: (start_datetime, end_datetime)
        step_size: seconds
    Returns:
        ephemeris: [N x 7] array (dt, x, y, z, vx, vy, vz)
        orbit_period_sec: orbital period [sec]
        keplerian: [N x 7] array (dt, sma, inc, ecc, raan, w, nu)
    """
    # Constants
    earth_GM = 398600.4415e9  # m^3/s^2
    earth_radius = 6378.137e3  # m
    j2 = 1.08263e-3

    # Convert angles to radians
    sma = sat['sma']*1e3  # Convert to km
    e = sat['ecc']
    i = np.radians(sat['inc'])
    raan = np.radians(sat['raan'])
    w = np.radians(sat['argp'])
    rad_ta = np.radians(sat['nu'])
    dt_epoch = sat['epoch']

    # Initial mean anomaly
    E1 = 2 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(rad_ta / 2))
    M0 = E1 - e * np.sin(E1)

    # Time steps (seconds since epoch)
    t_start = (dt_analysis_period[0] - dt_epoch).total_seconds()
    t_end = (dt_analysis_period[1] - dt_epoch).total_seconds()
    step = np.arange(t_start, t_end + step_size, step_size)

    # J2 rates
    n = np.sqrt(earth_GM / sma**3)
    orbit_period_sec = 2 * np.pi / n
    p = sma * (1 - e**2)
    factor = 1.5 * j2 * (earth_radius / p)**2 * n
    d_raan = -factor * np.cos(i)
    d_w = factor * (2 - 2.5 * np.sin(i)**2)
    d_m = factor * np.sqrt(1 - e**2) * (1 - 1.5 * np.sin(i)**2) + n

    ephemeris = []
    keplerian = []

    for this_dt in step:
        new_raan = raan + d_raan * this_dt
        new_w = w + d_w * this_dt
        new_m = M0 + d_m * this_dt

        # Solve Eccentric anomaly (E) by Newton-Raphson
        E = new_m
        for _ in range(1000):
            delta = (E - e * np.sin(E) - new_m) / (1 - e * np.cos(E))
            E -= delta
            if abs(delta) < 1e-8:
                break

        # True anomaly
        nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))
        r = sma * (1 - e * np.cos(E))

        # Position in orbital plane
        x_orb = r * np.cos(nu)
        y_orb = r * np.sin(nu)

        # Velocity in orbital plane
        p = sma * (1 - e**2)
        vx_orb = -np.sqrt(earth_GM / p) * np.sin(nu)
        vy_orb = np.sqrt(earth_GM / p) * (e + np.cos(nu))

        # Rotation matrices
        R3_w = np.array([
            [np.cos(new_w), -np.sin(new_w), 0],
            [np.sin(new_w),  np.cos(new_w), 0],
            [0, 0, 1]
        ])
        R1_i = np.array([
            [1, 0, 0],
            [0, np.cos(i), -np.sin(i)],
            [0, np.sin(i),  np.cos(i)]
        ])
        R3_RAAN = np.array([
            [np.cos(new_raan), -np.sin(new_raan), 0],
            [np.sin(new_raan),  np.cos(new_raan), 0],
            [0, 0, 1]
        ])
        R = R3_RAAN @ R1_i @ R3_w

        # Position and velocity in ECI
        r_teme = R @ np.array([x_orb, y_orb, 0])
        v_teme = R @ np.array([vx_orb, vy_orb, 0])

        ephemeris.append([dt_epoch + timedelta(seconds=this_dt), *r_teme, *v_teme])
        keplerian.append([this_dt, sma, np.degrees(i), e, np.degrees(new_raan), np.degrees(new_w), np.degrees(nu)])

    ephemeris = np.array(ephemeris)
    keplerian = np.array(keplerian)
    return ephemeris, orbit_period_sec, keplerian

# Example usage:
if __name__ == "__main__":
    from datetime import datetime, timedelta
    sma_m = 7000e3
    ecc = 0.001
    deg_inc = 98.7
    deg_raan = 0
    deg_w = 0
    deg_TA = 0
    dt_epoch = datetime(2025, 8, 21, 0, 0, 0)
    dt_analysis_period = (dt_epoch, dt_epoch + timedelta(hours=2))
    step_size = 60  # seconds

    eph, period, kep = propagator_J2(sma_m, ecc, deg_inc, deg_raan, deg_w, deg_TA, dt_epoch, dt_analysis_period, step_size)
    print(f"Orbit period: {period:.2f} sec")
    print("First ephemeris row:", eph[0])
    print("First keplerian row:", kep[0])