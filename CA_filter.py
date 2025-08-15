from datetime import datetime, timedelta
from TLEmanager import TLEmanger
from propagators import propagators
from orbitcalculator import orcal
from constants import constants as const
import numpy as np
import math

class CA_filter:
    def __init__(self):
        self.tle_manager = TLEmanger()
        self.propagator = propagators()
        self.orcal = orcal()  # Placeholder values for name, radius, and period
        pass

    def filter_outdated_tles(self, tle_data, start_time, end_time, pad_days):
        """
        Filters out TLEs that are older than 30 days from the current time.
        :param tle_data: List of tuples containing TLE data (epoch, line1, line2).
        :param current_time: Current datetime to compare against.
        :return: Filtered list of TLEs.
        """
        filtered_tles = []
        start_limit = start_time - timedelta(days=pad_days)
        end_limit = end_time + timedelta(days=pad_days)
        
        for norad, line1, line2, tle_epoch in tle_data:           
            if start_limit <= tle_epoch <= end_limit:
                filtered_tles.append((norad, line1, line2, tle_epoch))
        return filtered_tles
    
    def filter_altitude(self, tle_data, ref_line2, pad=0):
        filtered_tles = []
        ref_apogee, ref_perigee = self.tle_manager.compute_apogee_perigee(ref_line2)

        for i in range(len(tle_data)):
            norad, line1, line2, tle_epoch = tle_data[i]
            sat2_apogee, sat2_perigee = self.tle_manager.compute_apogee_perigee(line2)

            # Check if the apogee and perigee are within the specified pad
            if not((sat2_apogee < ref_perigee - pad) or (sat2_perigee > ref_apogee + pad)):
                filtered_tles.append((norad, line1, line2, tle_epoch))
        
        return filtered_tles
        
    def filter_orbitpath(self, tle_data, ref_line2, N_points=36, threshold=30):
        """
        Filters TLEs based on their orbit path.
        :param tle_data: List of tuples containing TLE data (norad, line1, line2, epoch) except reference sat.
        :param ref_line2: Reference TLE line2 for orbit path comparison.
        :param N_points: Number of points to sample along the orbit path.
        :param threshold: [km] Distance threshold for filtering.
        :return: Filtered list of TLEs.
        """
        filtered_tles = []
        N = np.linspace(0, 2 * np.pi, N_points)  # Sample points along the orbit path

        # Generate a reference orbit path using the reference satellite's TLE                        
        ref_orb = self.tle_manager.extract_elements(ref_line2)
        ref_r = self.propagator.orbit_path(ref_orb, N)  # Reference orbit path        
        
        # Iterate through the TLE except the reference TLE
        for i in range(len(tle_data)):
            norad, line1, line2, tle_epoch = tle_data[i]
            orbit = self.tle_manager.extract_elements(line2)

            # Compare the orbit with the reference orbit
            sat2_r = self.propagator.orbit_path(orbit, N)  # Assuming theta = 0 for initial comparison

            # Compare the two orbits
            distance = np.linalg.norm(ref_r[:, :, None] - sat2_r[:, None, :], axis=0)
            flag_ca = np.any(distance < threshold)         
            
            if flag_ca:
                filtered_tles.append((norad, line1, line2, tle_epoch))

        return filtered_tles
    
    # -------------------------
    # 6) filter_time (메인)
    # - coarse sampling over analysis_days with adaptive dt
    # - spawn candidate pass epochs (asc) inside each coarse window
    # - return list of candidate epochs for fine-check
    # -------------------------
    def filter_time(self, tle_data, ref_sat, time_window, analysis_days, d_tol_km=50.0):
        """
        returns:
        candidates: list of tuples (ref_epoch_candidate_datetime, sat2_epoch_candidate_datetime)
        Usage assumptions:
        - ref_sat is structure: (line0, line1, line2, epoch_datetime)
        - tle_data is list of other sats, here we use tle_data[0] as an example sat
        """
        # extract orbits
        ref_orbit = self.tle_manager.extract_elements(ref_sat[2], ref_sat[1])
        sat2_orbit = self.tle_manager.extract_elements(tle_data[0][2], tle_data[0][1])

        # compute secular rates
        ref_rates = self.compute_j2_rates(ref_orbit)
        sat2_rates = self.compute_j2_rates(sat2_orbit)

        # choose coarse dt (seconds)
        coarse_dt = self.choose_coarse_dt(orbit1=ref_orbit, orbit2=sat2_orbit, d_tol_km=d_tol_km)

        candidates = []

        total_seconds = 86400 * analysis_days
        t = 0.0
        while t < total_seconds:
            # propagate orbits to this coarse epoch (secular linear)
            ref_prop = self.propagate_orbit_j2(ref_orbit, ref_rates, t)
            sat2_prop = self.propagate_orbit_j2(sat2_orbit, sat2_rates, t)

            # line of nodes at this coarse epoch
            line = self.find_line_of_nodes(self.orcal, ref_prop['i'], ref_prop['Om'], sat2_prop['i'], sat2_prop['Om'])
            if line is not None:
                # compute time from coarse epoch to node pass for both sats
                try:
                    t1 = self.compute_node_time(ref_prop, line)
                    t2 = self.compute_node_time(sat2_prop, line)
                except Exception:
                    t1 = None
                    t2 = None

                if t1 is not None and t2 is not None:
                    # candidate epoch times (datetime)
                    ref_epoch_candidate = ref_sat[3] + timedelta(seconds=(t + t1))
                    sat2_epoch_candidate = tle_data[0][3] + timedelta(seconds=(t + t2))

                    # fast spatial filter: check whether node-longitude proximity is within window
                    # We'll create a coarse time window around the candidate (e.g., +/- time_window/2)
                    # and store for later fine filtering.
                    window_half = time_window / 2.0
                    candidates.append({
                        'ref_time': ref_epoch_candidate,
                        'sat2_time': sat2_epoch_candidate,
                        # 'coarse_center_t': t,
                        # 'ref_prop': ref_prop,
                        # 'sat2_prop': sat2_prop,
                        # 'dt_from_epoch_ref': t1,
                        # 'dt_from_epoch_sat2': t2,
                        # 'window_half_s': window_half
                    })

            t += coarse_dt

        # deduplicate / merge nearby candidates if desired (optional)
        # e.g., if two candidates' times are closer than some threshold, merge into one window
        # (left as exercise — but can be important to avoid duplicate fine-checks)

        return candidates
    
    # -------------------------
    # Helper: 단위 변환
    # -------------------------
    def revday_to_rad_s(self, n_rev_per_day):
        return n_rev_per_day * 2.0 * np.pi / 86400.0

    def revday2_to_rad_s2(self, n_dot_rev_day2):
        return n_dot_rev_day2 * 2.0 * np.pi / (86400.0**2)
    
    # -------------------------
    # 1) J2 secular rates 계산
    # 입력: orbit dict with keys 'a','e','i','n' (n: rev/day), optionally 'n_dot'
    # 출력: dict with dOm, dw, dotM_total (all in rad/s) and optionally n_dot_rad_s2
    # -------------------------
    def compute_j2_rates(self, orbit):
        # ensure n (rev/day) exists
        n_rad_s = self.revday_to_rad_s(orbit['n'])
        a = orbit['a']
        e = orbit['e']
        i = orbit['i']
        p = a * (1 - e**2)

        # common factor F = 3/2 * J2 * n * (Re/p)^2
        F = 1.5 * const.J2 * n_rad_s * (const.EARTH_RADIUS_KM**2 / p**2)

        dOm = -F * np.cos(i)  # rad/s
        dw = 0.5 * F * (5.0 * np.cos(i)**2 - 1.0)  # rad/s

        # mean anomaly secular total rate = n + correction
        dM_corr = (3.0/4.0) * const.J2 * n_rad_s * (const.EARTH_RADIUS_KM**2 / p**2) * np.sqrt(1 - e**2) * (3.0 * np.cos(i)**2 - 1.0)
        dotM_total = n_rad_s + dM_corr  # rad/s

        out = {
            'n_rad_s': n_rad_s,
            'dOm': dOm,
            'dw': dw,
            'dotM': dotM_total
        }

        if 'n_dot' in orbit and orbit['n_dot'] is not None:
            out['n_dot_rad_s2'] = self.revday2_to_rad_s2(orbit['n_dot'])
        return out

    # -------------------------
    # 2) 단순 J2 선형 propagate (secular만)
    # 입력: orbit dict, rates dict, dt (seconds)
    # 출력: new orbit dict (i, Om, w, ma updated)
    # -------------------------
    def propagate_orbit_j2(self, orbit, rates, dt):
        # If n_dot exists, include quadratic term into mean anomaly
        ma0 = orbit['ma']
        n_dot_term = 0.0
        if 'n_dot' in orbit and orbit['n_dot'] is not None:
            # convert n_dot to rad/s^2
            n_dot_rs2 = self.revday2_to_rad_s2(orbit['n_dot'])
            # if n_dot present, it affects mean motion over time -> contributes 0.5 * n_dot * t^2 to anomaly
            n_dot_term = 0.5 * n_dot_rs2 * dt**2

        ma_new = ma0 + rates['dotM'] * dt + n_dot_term 
        Om_new = orbit['Om'] + rates['dOm'] * dt
        w_new = orbit['w'] + rates['dw'] * dt

        return {
            'a': orbit['a'],
            'e': orbit['e'],
            'i': orbit['i'],
            'Om': Om_new,
            'w': w_new,
            'ma': ma_new % (2*np.pi),
            'n': orbit['n']  # keep original rev/day for period calc if needed
        }

    # -------------------------
    # 3) 노드 벡터 계산
    # -------------------------
    def find_line_of_nodes(self, orcal, i1, Om1, i2, Om2):
        n1 = orcal.orbit_normal_vector(i1, Om1)
        n2 = orcal.orbit_normal_vector(i2, Om2)
        line = np.cross(n1, n2)
        norm = np.linalg.norm(line)
        if norm < 1e-12:
            return None
        return line / norm
    
    # -------------------------
    # 4) node anomaly 및 node 통과 시간 계산 (asc only here, can extend to desc)
    # 입력: orbit(after propagate), line_of_nodes, orcal
    # 출력: t_node_seconds (float) - time from that epoch to node pass (sec)
    # -------------------------
    def compute_node_time(self, orbit_prop, line_of_nodes):
        # anomaly_at_node returns nu (rad)
        nu_asc = self.orcal.anomaly_at_node(orbit_prop['i'], orbit_prop['Om'], orbit_prop['w'], line_of_nodes)
        # time_to_anomaly expects e, n (rad/s), ma, nu (all in radians/n units)
        # n_rad_s = self.revday_to_rad_s(orbit_prop['n'])
        t_to_node = self.orcal.time_to_anomaly(orbit_prop['e'], orbit_prop['n'], orbit_prop['ma'], nu_asc)
        return float(t_to_node)
    
    # -------------------------
    # 5) 간격 결정 로직 (hybrid: period-based and relative-speed based)
    # 입력: orbit1, orbit2 dicts
    # 출력: dt_seconds (float) conservative lower-bound step
    # -------------------------
    def choose_coarse_dt(self, orbit1, orbit2, d_tol_km=100.0, min_dt=3600.0, max_dt=6*3600.0):
        # period based: T = 86400 / n_rev_day
        T1 = 86400.0 / orbit1['n']
        T2 = 86400.0 / orbit2['n']
        dt_period = min(T1, T2) / 4.0  # quarter of the faster satellite's period

        # velocity-based conservative estimate: use circular speed ~ sqrt(mu/a)
        v1 = math.sqrt(const.MU / orbit1['a'])
        v2 = math.sqrt(const.MU / orbit2['a'])
        v_rel_est = v1 + v2  # conservative upper bound
        if v_rel_est <= 0:
            dt_velocity = dt_period
        else:
            dt_velocity = d_tol_km / v_rel_est

        dt = min(dt_period, dt_velocity)
        # clamp
        dt = max(min_dt, min(dt, max_dt))
        return dt
        


from DBmanager import DBmanager

if __name__ == "__main__":
    CA_filter_instance = CA_filter()
    dbman = DBmanager()

    # Example usage of CA_filter methods
    sat1_tle_data = dbman.get_single_TLE(46025)  # Example NORAD number
    sat2_tle_data = dbman.get_single_TLE(63485)  # Example NORAD number

    res = CA_filter_instance.filter_time(sat2_tle_data, sat1_tle_data[0], 60, 10)  # Example time window in minutes


    pass