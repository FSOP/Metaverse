from datetime import datetime, timedelta
from TLEmanager import TLEmanger
from propagators import propagators
from orbitcalculator import orcal
from constants import constants as const
from sgp4.api import Satrec, jday
from math import acos, atan2, sqrt, pi
import numpy as np


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
            if not (
                (sat2_apogee < ref_perigee - pad) or (sat2_perigee > ref_apogee + pad)
            ):
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
            sat2_r = self.propagator.orbit_path(
                orbit, N
            )  # Assuming theta = 0 for initial comparison

            # Compare the two orbits
            distance = np.linalg.norm(ref_r[:, :, None] - sat2_r[:, None, :], axis=0)
            flag_ca = np.any(distance < threshold)

            if flag_ca:
                filtered_tles.append((norad, line1, line2, tle_epoch))

        return filtered_tles

    # -------------------------
    # angle helpers
    # -------------------------
    def wrap_2pi(self, x):
        y = np.mod(x, 2 * np.pi)
        return y if y >= 0.0 else y + 2 * np.pi

    def angle_between_unit(self, a, b):
        # both unit; robust acos clamp
        c = float(np.dot(a, b))
        c = max(-1.0, min(1.0, c))
        return acos(c)

    # -------------------------
    # TLE -> Satrec
    # -------------------------
    def satrec_from_tle(self, line1, line2):
        return Satrec.twoline2rv(line1, line2)

    # -------------------------
    # SGP4 propagate to a datetime => r (km), v (km/s)
    # times are UTC
    # -------------------------
    def sgp4_rv_at(self, satrec: Satrec, when: datetime):
        jd, fr = jday(
            when.year,
            when.month,
            when.day,
            when.hour,
            when.minute,
            when.second + when.microsecond / 1e6,
        )
        e, r, v = satrec.sgp4(jd, fr)
        if e != 0:
            raise RuntimeError(f"SGP4 error code {e}")
        return np.array(r, dtype=float), np.array(v, dtype=float)

    # -------------------------
    # choose adaptive coarse dt for scanning (hybrid)
    # -------------------------
    def choose_coarse_dt_by_period_and_speed(
        self, a1, a2, d_tol_km=100.0, min_dt=60.0, max_dt=6 * 3600.0
    ):
        n1 = sqrt(const.MU / (a1**3))
        n2 = sqrt(const.MU / (a2**3))
        T1 = 2 * np.pi / n1
        T2 = 2 * np.pi / n2
        dt_period = min(T1, T2) / 4.0  # faster sat 기준 1/4 주기

        v1 = sqrt(const.MU / a1)
        v2 = sqrt(const.MU / a2)
        v_rel_est = v1 + v2
        dt_speed = d_tol_km / v_rel_est

        dt = min(dt_period, dt_speed)
        dt = max(min_dt, min(dt, max_dt))
        return dt

    # -------------------------
    # 메인 pre-filter (SGP4 기반, asc+desc)
    #   - ref_sat: (name?, line1, line2, epoch_datetime)
    #   - other   : (name?, line1, line2, epoch_datetime)  # 여긴 예시로 tle_data[0]
    #   - analysis_days: 30 등
    #   - time_window: fine 단계로 넘길 윈도 폭(초) (여기선 meta만 저장)
    # 반환: candidate dict 리스트
    # -------------------------
    def filter_time(
        self, ref_sat, tle_data, analysis_days=10, time_window=600.0, d_tol_km=100.0
    ):
        _, ref_l1, ref_l2, ref_epoch = ref_sat

        candidates = []
        # 시작 시각: 공통 기준(여기서는 ref_epoch)
        start = ref_epoch
        end = start + timedelta(days=analysis_days)

        # ref_epoch 기준으로 함
        sat_ref = self.satrec_from_tle(ref_l1, ref_l2)
        r1, v1 = self.sgp4_rv_at(sat_ref, start)
        el1 = self.orcal.elements_from_rv(r1, v1)

        #
        for other_sat in tle_data:
            _, o_l1, o_l2, o_epoch = other_sat
            sat_o = self.satrec_from_tle(o_l1, o_l2)
            r2, v2 = self.sgp4_rv_at(sat_o, start)
            el2 = self.orcal.elements_from_rv(r2, v2)

            coarse_dt = self.choose_coarse_dt_by_period_and_speed(
                el1["a"], el2["a"], d_tol_km=d_tol_km, min_dt=3600
            )

            reference_interval = []
            candidate_interval = []
            t = start
            while t < end:
                try:
                    r1, v1 = self.sgp4_rv_at(sat_ref, t)
                    r2, v2 = self.sgp4_rv_at(sat_o, t)
                except RuntimeError:
                    t += timedelta(seconds=coarse_dt)
                    continue

                el1 = self.orcal.elements_from_rv(r1, v1)
                el2 = self.orcal.elements_from_rv(r2, v2)

                # 공통 교차선
                line_u = self.orcal.common_line_of_nodes(el1["h_hat"], el2["h_hat"])
                if line_u is None:
                    t += timedelta(seconds=coarse_dt)
                    continue

                # asc/desc 두 방향 모두 시도
                for asc_flag in (True, False):
                    nu1 = self.orcal.target_true_anomaly_for_line(
                        line_u, el1, asc=asc_flag
                    )
                    nu2 = self.orcal.target_true_anomaly_for_line(
                        line_u, el2, asc=asc_flag
                    )
                    if nu1 is None or nu2 is None:
                        continue

                    dt1 = self.orcal.time_to_reach_true_anomaly(el1, nu1)
                    dt2 = self.orcal.time_to_reach_true_anomaly(el2, nu2)
                    if dt1 is None or dt2 is None:
                        continue

                    # 후보 시각 (각 위성의 다음 교차선 통과 시각)
                    cand_ref_time = t + timedelta(seconds=dt1)
                    cand_o_time = t + timedelta(seconds=dt2)

                    if dt1 < coarse_dt:
                        reference_interval.append(cand_ref_time)
                    if dt2 < coarse_dt:
                        candidate_interval.append(cand_o_time)

                t += timedelta(seconds=coarse_dt)

            # 후보 시각 정리
            for ref_time in reference_interval:
                for cand_time in candidate_interval:
                    if abs((cand_time - ref_time).total_seconds()) <= time_window:
                        candidates.append(
                            {
                                "ref_time": ref_time,
                                "cand_time": cand_time,
                                "ref_sat": ref_sat,
                                "cand_sat": other_sat,
                            }
                        )
        return candidates

    def fine_filter_min_distance(self, ref_sat, other_sats, candidates, dt_s=1.0):
        """
        ref_sat, other_sat: (name?, line1, line2, epoch)
        candidates: pre-filter 결과 리스트
        dt_s: fine-sampling 간격 (초)

        반환: list of dict {'type','ref_time','other_time','closest_distance_km','closest_time'}
        """
        results = []
        # SGP4
        sat_ref = Satrec.twoline2rv(ref_sat[1], ref_sat[2])

        for cand in candidates:
            sat_o = Satrec.twoline2rv(cand["cand_sat"][1], cand["cand_sat"][2])
            n_steps = 600  # 예시로 10분 간격으로 샘플링
            times = min(cand["ref_time"], cand["cand_time"]) - timedelta(minutes=1)  # 기준 시각을 중심으로
            times = [times + timedelta(seconds=i * dt_s) for i in range(n_steps)]

            min_dist = np.inf
            t_min = None

            for t in times:
                # SGP4 propagate
                jd, fr = jday(
                    t.year,
                    t.month,
                    t.day,
                    t.hour,
                    t.minute,
                    t.second + t.microsecond / 1e6,
                )
                e1, r1, _ = sat_ref.sgp4(jd, fr)
                e2, r2, _ = sat_o.sgp4(jd, fr)
                if e1 != 0 or e2 != 0:
                    continue

                r1 = np.array(r1)
                r2 = np.array(r2)
                d = np.linalg.norm(r1 - r2)
                if d < min_dist:
                    min_dist = d
                    t_min = t

            if t_min is not None and min_dist < 10.0:  # 예시로 100km 이하
                results.append(
                    {
                        # 'type': cand['type'],
                        "sat1_norad": ref_sat[1][2:7].strip(),  # Extract NORAD number from line1
                        "sat2_norad": cand["cand_sat"][1][2:7].strip(),  # Extract NORAD number from line1
                        "ref_time": cand["ref_time"],
                        "other_time": cand["cand_time"],
                        "closest_distance_km": min_dist,
                        "closest_time": t_min,
                    }
                )
        return results


from DBmanager import DBmanager

if __name__ == "__main__":
    CA_filter_instance = CA_filter()
    dbman = DBmanager()

    # Example usage of CA_filter methods
    sat1_tle_data = dbman.get_single_TLE(46025)  # Example NORAD number
    sat2_tle_data = dbman.get_single_TLE(63485)  # Example NORAD number

    res = CA_filter_instance.filter_time(
        sat1_tle_data[0], sat2_tle_data[0], 60, 10
    )  # Example time window in minutes

    pass
