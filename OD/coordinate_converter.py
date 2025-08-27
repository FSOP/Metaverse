import os, sys 
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_root)

import numpy as np
from datetime import datetime
from astropy.time import Time
from HPOP.coordinate_systems import compute_iers_matrices
from HPOP.eop import EOPManager
from HPOP.time_utils import convert_time_scales
from collections import namedtuple

class coordinate_converter:
    def __init__(self):
        self.eop_manager = EOPManager()
        # self.time_utils = convert_time_scales()
        pass

    def los_vectors(self, meas):
        """
        Calculate Line-Of-Sight unit vectors from [RA, Dec] observations.
        Args:
            meas: (n,2) array of [RA, Dec] in degrees
        Returns:
            LOS: (3, n) array, each column is a LOS unit vector
        """
        meas = np.asarray(meas)
        n = meas.shape[0]
        LOS = np.zeros((3, n))
        for i in range(n):
            RA = np.radians(meas[i, 0])
            dec = np.radians(meas[i, 1])
            LOS[:, i] = [np.cos(dec)*np.cos(RA), np.cos(dec)*np.sin(RA), np.sin(dec)]
        return LOS
            
    def cartesian_to_keplerian(self, r, v=None, GM=3.986004415e14):
        """
        Compute Keplerian orbital elements from Cartesian states.
        Args:
            r: position vector (3,) or state vector (6,)
            v: velocity vector (3,), optional if r is 6 elements
            GM: gravitational parameter (default: EGM-96 value, m^3/s^2)
        Returns:
            namedtuple('KeplerianElements', ['sma', 'ecc', 'incl', 'raan', 'argp', 'tran'])
            All angles in radians, SMA in units of r and GM.
        """
        KeplerianElements = namedtuple('KeplerianElements', ['sma', 'ecc', 'incl', 'raan', 'argp', 'tran'])
        r = np.asarray(r).flatten()
        if v is None:
            if r.size != 6:
                raise ValueError("If v is not provided, r must be a 6-element state vector.")
            v = r[3:6]
            r = r[0:3]
        else:
            v = np.asarray(v).flatten()
            if r.size != 3 or v.size != 3:
                raise ValueError("r and v must be 3-element vectors.")
    
        k = np.array([0, 0, 1])
        h = np.cross(r, v)
        n = np.cross(k, h)
        N = np.linalg.norm(n)
        H2 = np.dot(h, h)
        V2 = np.dot(v, v)
        R = np.linalg.norm(r)
        e_vec = ((V2 - GM/R) * r - np.dot(r, v) * v) / GM
        p = H2 / GM
    
        ecc = np.linalg.norm(e_vec)
        sma = p / (1 - ecc**2)
        incl = np.arccos(h[2] / np.sqrt(H2))
        raan = np.arccos(n[0] / N) if N > 0 else 0.0
        if n[1] < 0:
            raan = 2 * np.pi - raan
        argp = np.arccos(np.dot(n, e_vec) / (N * ecc)) if N > 0 and ecc > 0 else 0.0
        if e_vec[2] < 0:
            argp = 2 * np.pi - argp
        tran = np.arccos(np.dot(e_vec, r) / (ecc * R)) if ecc > 0 else 0.0
        if np.dot(r, v) < 0:
            tran = 2 * np.pi - tran
    
        return KeplerianElements(sma, ecc, incl, raan, argp, tran)
       
    def aer_to_radec(self, az, el, site, epoch):
        """
        Convert azimuth/elevation to right ascension/declination using ENU→ECEF→ICRF transformation.
        Args:
            az, el: Azimuth, Elevation in degrees
            obs_site: [lat_deg, lon_deg, alt_m] observer geodetic coordinates
            epoch: datetime object (UTC)
        Returns:
            (RA, Dec) in degrees
        """
        # obs_lat, obs_lon, obs_alt = [float(s) for s in obs_site]        
        ra, dec = [], []
        site_lon = site['lon']
        site_lat = site['lat']
        LTCs, Us = [], []

        for this_az, this_el, this_epoch in zip(az, el, epoch):
            # 1. AER to ENU unit vector
            enu_vector = self.aer_to_enz(this_az, this_el)

            # 2. ENU to ECEF direction vector
            LTC = self.LTCMatrix(site_lon, site_lat)  # Local Tangent Plane to ECEF rotation matrix
            ecef_dir = LTC.T @ enu_vector  # ENU to ECEF direction

            # 3. ECEF to ICRF (ECI/J2000)
            mjd_utc = Time(this_epoch, scale='utc').mjd
            x_pole, y_pole, _, _, _, _, _, _, _ = self.eop_manager.get_eop_values(mjd_utc)
            mjd_ut1, mjd_tt, mjd_tdb = convert_time_scales(mjd_utc, self.eop_manager)

            U = compute_iers_matrices(mjd_tt, mjd_ut1, x_pole, y_pole)
            icrf_vec = U.T @ ecef_dir  # ECEF to ICRF
            # 4. ICRF unit vector to RA/Dec
            icrf_unit = icrf_vec / np.linalg.norm(icrf_vec)
            dec.append(np.degrees(np.arcsin(icrf_unit[2])))
            ra.append(np.degrees(np.arctan2(icrf_unit[1], icrf_unit[0])) % 360)
            LTCs.append(LTC)
            Us.append(U)

        return ra, dec, LTCs, Us

    def enz_to_icrf(self, enz_vec, obs_site, epoch):
        """
        Convert ENZ (East-North-Zenith) vector at observer site to ICRF (ECI/J2000) frame.
        Args:
            enz_vec: ENZ vector (3,)
            obs_site: [lat_deg, lon_deg, alt_m]
            epoch: datetime object (UTC)
        Returns:
            ICRF vector (3,)
        """
        obs_lat, obs_lon, obs_alt = obs_site
        lat = np.radians(obs_lat)
        lon = np.radians(obs_lon)
        
        R = np.array([
            [-np.sin(lon),              np.cos(lon),               0],
            [-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon),  np.cos(lat)],
            [ np.cos(lat)*np.cos(lon),  np.cos(lat)*np.sin(lon),  np.sin(lat)]
        ])
        ecef_dir = R.T @ enz_vec
        mjd_utc = Time(epoch, scale='utc').mjd
        x_pole, y_pole, ut1_utc, lod, dpsi, deps, dx_pole, dy_pole, tai_utc = self.eop_manager.get_eop_values(mjd_utc)
        mjd_ut1, mjd_tt, mjd_tdb = convert_time_scales(mjd_utc, self.eop_manager)
        E = compute_iers_matrices(mjd_tt, mjd_ut1, x_pole, y_pole)
        icrf_vec = E @ ecef_dir

        return icrf_vec

    def eci_to_ecef(self, r_eci, epoch):
        """
        Convert ECI (TEME) position to ECEF at given epoch.
        r_eci: [x, y, z] in meters
        epoch: datetime object (UTC)
        Returns: [x, y, z] in meters (ECEF)
        """
        # Compute GMST (Greenwich Mean Sidereal Time) in radians
        mjd_utc = Time(epoch, scale='utc').mjd

        x_pole, y_pole, ut1_utc, lod, dpsi, deps, dx_pole, dy_pole, tai_utc = \
            self.eop_manager.get_eop_values(mjd_utc)        
        
        mjd_ut1, mjd_tt, mjd_tdb = convert_time_scales(mjd_utc, self.eop_manager)

        E = compute_iers_matrices(mjd_tt, mjd_ut1, x_pole, y_pole)
        E_transpose = E.T

        return E @ r_eci
    
    def ecef_to_eci(self, r_ecef, epoch):
        """
        Convert ECEF (Earth-Centered Earth-Fixed) position to ECI (Earth-Centered Inertial) at given epoch.
        r_ecef: [x, y, z] in meters
        epoch: datetime object (UTC)
        Returns: [x, y, z] in meters (ECI)
        """
        # Compute GMST (Greenwich Mean Sidereal Time) in radians
        mjd_utc = Time(epoch, scale='utc').mjd

        x_pole, y_pole, ut1_utc, lod, dpsi, deps, dx_pole, dy_pole, tai_utc = \
            self.eop_manager.get_eop_values(mjd_utc)

        mjd_ut1, mjd_tt, mjd_tdb = convert_time_scales(mjd_utc, self.eop_manager)

        E = compute_iers_matrices(mjd_tt, mjd_ut1, x_pole, y_pole)        

        return E.T @ r_ecef

    def geodetic_to_ecef(self, lat_deg, lon_deg, alt_m):
        # WGS-84 parameters
        a = 6378137.0
        e2 = 6.69437999014e-3
        lat = np.radians(lat_deg)
        lon = np.radians(lon_deg)
        N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
        x = (N + alt_m) * np.cos(lat) * np.cos(lon)
        y = (N + alt_m) * np.cos(lat) * np.sin(lon)
        z = (N * (1 - e2) + alt_m) * np.sin(lat)
        return np.array([x, y, z])

    def ecef_to_enu(self, r_ecef, obs_ecef, lat_deg, lon_deg):
        # ENU rotation matrix
        lat = np.radians(lat_deg)
        lon = np.radians(lon_deg)
        R = np.array([
            [-np.sin(lon),              np.cos(lon),               0],
            [-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon),  np.cos(lat)],
            [ np.cos(lat)*np.cos(lon),  np.cos(lat)*np.sin(lon),  np.sin(lat)]
        ])
        return R @ (r_ecef - obs_ecef)

    def aer_from_enu(self, enu):
        east, north, up = enu
        rng = np.linalg.norm(enu)
        az = np.degrees(np.arctan2(east, north)) % 360
        el = np.degrees(np.arcsin(up / rng))
        return az, el, rng

    def spherical_to_cartesian(self, az, el, r):
        """
        Convert spherical coordinates (azimuth, elevation, radius) to Cartesian (x, y, z).
        Azimuth and elevation should be in degrees, radius in desired units (e.g., meters).
        Azimuth is measured from x-axis (East), elevation from horizon (Up).
        Returns a numpy array [x, y, z].
        """
        az = np.radians(az)
        el = np.radians(el)
        x = r * np.cos(el) * np.cos(az)
        y = r * np.cos(el) * np.sin(az)
        z = r * np.sin(el)
        return np.array([x, y, z])

    def cartesian_to_spherical(self, cartesian):
        """
        Convert Cartesian coordinates (x, y, z) to spherical (azimuth, elevation, radius).
        Returns azimuth and elevation in degrees, radius in same units as input.
        Azimuth is measured from x-axis (East), elevation from horizon (Up).
        """
        x, y, z = cartesian
        r = np.linalg.norm(cartesian)
        el = np.degrees(np.arcsin(z / r))
        az = np.degrees(np.arctan2(y, x)) % 360
        return az, el, r

    def aer_to_enz(self, az, el):
        """
        Convert AER (azimuth, elevation, range) to ENZ (East-North-Zenith) vector.
        Useful for transforming ground-based observation angles to local Cartesian coordinates.
        Input: aer = [azimuth (deg), elevation (deg), range]
        Output: ENZ vector as numpy array [East, North, Zenith]
        """
        az = az * (np.pi / 180)
        el = el * (np.pi / 180)

        enz = [
                np.sin(az) * np.cos(el),  # East
                np.cos(az) * np.cos(el),  # North
                np.sin(el)                # Zenith
            ]
        return enz
    
    def LTCMatrix(self, lon, lat):
        lon, lat = np.radians(lon), np.radians(lat)
        # 각도는 radian 단위
        R_y = np.array([
            [np.cos(-lat), 0, -np.sin(-lat)],
            [0, 1, 0],
            [np.sin(-lat), 0, np.cos(-lat)]
        ])
        R_z = np.array([
            [np.cos(lon), np.sin(lon), 0],
            [-np.sin(lon),  np.cos(lon), 0],
            [0, 0, 1]
        ])
        M = R_y @ R_z
        # 행 순환
        for j in range(3):
            Aux = M[0, j]
            M[0, j] = M[1, j]
            M[1, j] = M[2, j]
            M[2, j] = Aux
        return M

    # Example usage:
    if __name__ == "__main__":
        # Satellite ECI position (meters)
        r_eci = np.array([7000e3, 0, 0])
        epoch = datetime(2025, 8, 21, 12, 0, 0)
        # Observer geodetic position
        lat_deg, lon_deg, alt_m = 37.5, 127.0, 100.0
        obs_ecef = geodetic_to_ecef(lat_deg, lon_deg, alt_m)
        # Convert satellite ECI to ECEF
        r_ecef = eci_to_ecef(r_eci, epoch)
        # Convert to ENU
        enu = ecef_to_enu(r_ecef, obs_ecef, lat_deg, lon_deg)
        # Calculate azimuth, elevation, range
        az, el, rng = aer_from_enu(enu)
        print(f"Azimuth: {az:.2f} deg, Elevation: {el:.2f} deg, Range: {rng/1000:.2f} km")