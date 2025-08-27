"""
Gauss.py

Angles-only Initial Orbit Determination methods:
- Gauss
- Gibbs
- Herrick-Gibbs

All functions assume NumPy arrays for vector/matrix math.
"""

import numpy as np
from OD.coordinate_converter import coordinate_converter

class Gauss:
    def __init__(self):
        self.cc = coordinate_converter()

    def gauss(self, meas, time, rsite, mu):
        """
        Gauss method for angles-only initial orbit determination.
        Args:
            meas: (3,2) array of [RA, Dec] in degrees
            time: (3,6) array of datevectors (year, month, day, hour, min, sec)
            rsite: (3,3) array of site position vectors in inertial frame (km)
            mu: gravitational parameter (km^3/s^2)
        Returns:
            sat_Pos: position vector at second observation (km)
            sat_Vel: velocity vector at second observation (km/s)
            Iterations: number of iterations to reach convergence
            r1: position vector at first observation (km)
            r3: position vector at third observation (km)
        """
        # LOS vectors
        LOS = self.cc.los_vectors(meas)  # shape (3,3), columns are LOS vectors

        # Time differences (seconds)
        t1 = self.etime(time[0], time[1])
        t3 = self.etime(time[2], time[1])

        # Coefficients
        a1 = t3 / (t3 - t1)
        a1u = t3 * ((t3 - t1)**2 - t3**2) / (6 * (t3 - t1))
        a3 = -t1 / (t3 - t1)
        a3u = -t1 * ((t3 - t1)**2 - t1**2) / (6 * (t3 - t1))

        M = np.linalg.inv(LOS) @ rsite

        d1 = M[1,0]*a1 - M[1,1] + M[1,2]*a3
        d2 = M[1,0]*a1u + M[1,2]*a3u

        C = np.dot(LOS[:,1], rsite[:,1])

        q1 = -(d1**2 + 2*C*d1 + np.dot(rsite[:,1], rsite[:,1]))
        q2 = -2*mu*(C*d2 + d1*d2)
        q3 = -mu**2 * d2**2

        # Solve 8th order polynomial
        coeffs = [1, 0, q1, 0, 0, q2, 0, 0, q3]
        r2_roots = np.roots(coeffs)
        r2_real = r2_roots[np.isreal(r2_roots)].real
        r2_real_pos = r2_real[r2_real > 0]
        if len(r2_real_pos) == 0:
            raise ValueError("No positive, real roots found. No solution.")
        r2 = r2_real_pos[0]

        u = mu / r2**3

        c1 = a1 + a1u*u
        c2 = -1
        c3 = a3 + a3u*u
        c = np.array([c1, c2, c3])

        cp = -1 * (M @ c)
        rhonot = np.array([cp[0]/c1, cp[1]/c2, cp[2]/c3])
        rho = rhonot.copy()
        rho_next = np.zeros(3)

        # Initial position vectors estimate
        r = np.column_stack([rhonot[i]*LOS[:,i] + rsite[:,i] for i in range(3)])

        i = 0
        Error = rhonot[0]*1e-6
        error_count = 0

        while (np.abs(rho[0]-rho_next[0]) > Error and
            np.abs(rho[1]-rho_next[1]) > Error and
            np.abs(rho[2]-rho_next[2]) > Error):

            alpha_12 = np.degrees(np.arccos(np.dot(r[:,0], r[:,1]) / (np.linalg.norm(r[:,0]) * np.linalg.norm(r[:,1]))))
            alpha_23 = np.degrees(np.arccos(np.dot(r[:,1], r[:,2]) / (np.linalg.norm(r[:,1]) * np.linalg.norm(r[:,2]))))

            # Check for coplanarity
            # If the angles are significantly different, we can use the Gibbs method
            if abs(alpha_12) > 3 and abs(alpha_23) > 3:
                v2, e, p = self.gibbs(r, mu)
            else: # Herrick-Gibbs 
                v2 = self.herrick_gibbs(r, time)
                KOE = self.cc.cartesian_to_keplerian(r[:,1], v2, mu)
                p = KOE['sma'] * (1 - KOE['ecc']**2)

            h = np.cross(r[:,1], v2)
            n = np.cross([0,0,1], h)
            e_vec = ((np.dot(v2, v2) - mu/np.linalg.norm(r[:,1]))*r[:,1] - np.dot(r[:,1], v2)*v2)/mu
            sme = np.dot(v2, v2)/2 - mu/np.linalg.norm(r[:,1])

            if np.linalg.norm(e_vec) == 1.0:
                p = np.dot(h, h)/mu
            else:
                a = -mu/(2*sme)
                p = a*(1-np.dot(e_vec, e_vec))

            f1 = 1 - (np.linalg.norm(r[:,0]) / p)*(1-np.cos(np.radians(-alpha_12)))
            f3 = 1 - (np.linalg.norm(r[:,2]) / p)*(1-np.cos(np.radians(alpha_23)))
            g1 = (np.linalg.norm(r[:,0])*np.linalg.norm(r[:,1])*np.sin(np.radians(-alpha_12)))/np.sqrt(mu*p)
            g3 = (np.linalg.norm(r[:,2])*np.linalg.norm(r[:,1])*np.sin(np.radians(alpha_23)))/np.sqrt(mu*p)
            c1 = g3 / (f1*g3 - f3*g1)
            c3 = -g1 / (f1*g3 - f3*g1)

            c = np.array([c1, c2, c3])
            cp = M @ (-1 * c)
            rho_temp = np.array([cp[0]/c1, cp[1]/c2, cp[2]/c3])

            if i == 1:
                rho = rho
                rho_next = rho_temp
            else:
                rho = rho_next
                rho_next = rho_temp

            i += 1
            r = np.column_stack([rho_next[j]*LOS[:,j] + rsite[:,j] for j in range(3)])
            if i >= 500:
                Error *= 10
                error_count += 1
                i = 0
                r = np.column_stack([rhonot[j]*LOS[:,j] + rsite[:,j] for j in range(3)])
                print("Increasing error tolerance by factor of 10.")
            if error_count == 6:
                print("WARNING: Solution diverged. Stopped at 3000 iterations.")
                return np.nan, np.nan, i, np.nan, np.nan

        sat_Pos = r[:,1]
        sat_Vel = v2
        r1 = r[:,0]
        r3 = r[:,2]
        Iterations = i
        return sat_Pos, sat_Vel, Iterations, r1, r3

    def gibbs(self, r, mu):
        """
        Gibbs method for velocity determination given three position vectors.
        Args:
            r: 3x3 numpy array, columns are position vectors at three times (km)
            mu: gravitational parameter (km^3/s^2)
        Returns:
            v2: velocity vector at second time (km/s)
            e: eccentricity estimate
            P: semilatus rectum (km)
        """
        r1 = r[:, 0]
        r2 = r[:, 1]
        r3 = r[:, 2]

        Z12 = np.cross(r1, r2)
        Z23 = np.cross(r2, r3)
        Z31 = np.cross(r3, r1)

        # Coplanarity check (~3 deg)
        a_COP = 90 - np.degrees(np.arccos(np.dot(Z23, r1) / (np.linalg.norm(Z23) * np.linalg.norm(r1))))
        if abs(a_COP) > 3:
            print(f"WARNING: Position vectors are not coplanar to 3 degree tolerance! Coplanar angle: {a_COP:.2f} deg")

        alpha_12 = np.degrees(np.arccos(np.dot(r1, r2) / (np.linalg.norm(r1) * np.linalg.norm(r2))))
        alpha_23 = np.degrees(np.arccos(np.dot(r2, r3) / (np.linalg.norm(r2) * np.linalg.norm(r3))))
        if abs(alpha_12) <= 1 or abs(alpha_23) <= 1:
            print(f"WARNING: Angle between position vectors is smaller than 1 degree. "
                f"alpha_12: {alpha_12:.2f}, alpha_23: {alpha_23:.2f}")

        N = np.linalg.norm(r1) * Z23 + np.linalg.norm(r2) * Z31 + np.linalg.norm(r3) * Z12
        D = Z12 + Z23 + Z31
        S = (np.linalg.norm(r2) - np.linalg.norm(r3)) * r1 + \
            (np.linalg.norm(r3) - np.linalg.norm(r1)) * r2 + \
            (np.linalg.norm(r1) - np.linalg.norm(r2)) * r3
        B = np.cross(D, r2)

        Lg = np.sqrt(mu / (np.linalg.norm(N) * np.linalg.norm(D)))

        # Direction of eccentricity (e_hat) used to find the true anomaly
        W_hat = N / np.linalg.norm(N)
        Q_hat = S / np.linalg.norm(S)
        e_hat = np.cross(Q_hat, W_hat)

        v2 = (Lg / np.linalg.norm(r2)) * B + Lg * S
        e = np.linalg.norm(S) / np.linalg.norm(D)
        P = np.linalg.norm(N) * e / np.linalg.norm(S)
        return v2, e, P

    def herrick_gibbs(self, r, time, mu):
        """
        Herrick-Gibbs method for velocity determination given three position vectors.
        Args:
            r: 3x3 numpy array, columns are position vectors at three times (km)
            time: 3x6 numpy array, rows are [year, month, day, hour, min, sec]
            mu: gravitational parameter (km^3/s^2)
        Returns:
            v2: velocity vector at second time (km/s)
        """
        r1 = r[:, 0]
        r2 = r[:, 1]
        r3 = r[:, 2]

        Z12 = np.cross(r1, r2)
        Z23 = np.cross(r2, r3)
        Z31 = np.cross(r3, r1)

        # Coplanarity check (~3 deg)
        a_COP = 90 - np.degrees(np.arccos(np.dot(Z23, r1) / (np.linalg.norm(Z23) * np.linalg.norm(r1))))
        if abs(a_COP) > 3:
            print(f"WARNING: Position vectors are not coplanar to 3 degree tolerance! Coplanar angle: {a_COP:.2f} deg")

        alpha_12 = np.degrees(np.arccos(np.dot(r1, r2) / (np.linalg.norm(r1) * np.linalg.norm(r2))))
        alpha_23 = np.degrees(np.arccos(np.dot(r2, r3) / (np.linalg.norm(r2) * np.linalg.norm(r3))))
        if abs(alpha_12) > 5 or abs(alpha_23) > 5:
            print(f"WARNING: Angle between position vectors is greater than 5 degrees. "
                f"alpha_12: {alpha_12:.2f}, alpha_23: {alpha_23:.2f}")

        # Time differences (seconds)
        t21 = self.etime(time[1], time[0])
        t32 = self.etime(time[2], time[1])
        t31 = self.etime(time[2], time[0])

        # Velocity estimation using Taylor Series expansion
        v2 = (-t32 * (1/(t21*t31) + mu/(12*np.linalg.norm(r1)**3)) * r1 +
            (t32 - t21) * (1/(t21*t32) + mu/(12*np.linalg.norm(r2)**3)) * r2 +
            t21 * (1/(t32*t31) + mu/(12*np.linalg.norm(r3)**3)) * r3)
        return v2

    def etime(self, t1, t2):
        """
        Returns the time difference in seconds between two datevectors or datetime objects.
        t1, t2: either arrays [year, month, day, hour, min, sec] or datetime.datetime objects
        """
        from datetime import datetime
        # If t1/t2 are datetime objects, use directly
        if hasattr(t1, 'year') and hasattr(t1, 'month') and hasattr(t1, 'day'):
            dt1 = t1
        else:
            dt1 = datetime(*map(int, t1))
        if hasattr(t2, 'year') and hasattr(t2, 'month') and hasattr(t2, 'day'):
            dt2 = t2
        else:
            dt2 = datetime(*map(int, t2))
        return (dt1 - dt2).total_seconds()
    # You can add any helper functions (e.g., los_vectors, kepel) below as needed.