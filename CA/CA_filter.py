from datetime import datetime, timedelta
from MISC.TLEmanager import TLEmanager
from CA.propagators import propagators
from CA.orbitcalculator import orcal
from sgp4.api import Satrec, jday
from math import acos, atan2, sqrt, pi

import MISC.constants as const
import numpy as np


class CA_filter:
    def __init__(self):
        self.tle_manager = TLEmanager()
        self.propagator = propagators()
        self.orcal = orcal()  # Placeholder values for name, radius, and period
        self.critera = {
            'minium_distance': 10.0,  # km  
        }
        pass

    def filter_BSTAR(self, tle, threshold_bstar):
        """
        Filters TLEs based on B* (drag term).
        :param tle: TLE data
        :param threshold_bstar: B* threshold
        :return: Filtered TLEs
        """
        filtered_tles = []
        for t in tle:
            bstar = self.tle_manager.extract_bstar(t[2])
            if abs(bstar) > threshold_bstar:
                filtered_tles.append(t)
        return filtered_tles
    

    def filter_perigee(self, tle, threshold_alt):
        """  
        Filters TLEs based on perigee altitude.
        :param tle: TLE data
        :param threshold_alt: Altitude threshold
        :return: Filtered TLEs
        """
        filtered_tles = []
        for t in tle:
            _, perigee = self.tle_manager.compute_apogee_perigee(t[2])
            if perigee < threshold_alt:
                filtered_tles.append(t)
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
                    # ref_time: 기준 위성이 교차선(공통 궤도 평면)을 통과하는 예상 시각
                    # cand_time: 후보 위성이 같은 교차선을 통과하는 예상 시각
                    # 두 값의 차이가 time_window 이내면 근접 가능성이 있다고 판단
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
        주어진 후보 위성 쌍(candidates)에 대해, 기준(ref) 위성과의 최소 접근 거리를 세밀하게 계산하는 함수.

        Args:
            ref_sat: 기준 위성의 TLE 정보 튜플 (name, line1, line2, epoch)
            other_sats: 비교 대상 위성 리스트 (사용하지 않음, candidates에 포함됨)
            candidates: coarse filter를 통과한 후보 쌍 리스트. 각 원소는 dict로, 'cand_sat'에 TLE, 'ref_time', 'cand_time' 포함.
            dt_s: 샘플링 간격(초). 최소 접근 거리 탐색 시 시간 간격.

        Returns:
            list of dict. 각 dict는 다음 정보를 포함:
                'sat1_norad': 기준 위성의 NORAD 번호
                'sat2_norad': 상대 위성의 NORAD 번호
                'ref_time': 기준 위성의 기준 시각
                'other_time': 상대 위성의 기준 시각
                'closest_distance_km': 최소 접근 거리 (km)
                'closest_time': 최소 접근 거리 발생 시각
        """
        results = []
        # 기준 위성 SGP4 객체 생성
        sat_ref = Satrec.twoline2rv(ref_sat[1], ref_sat[2])

        for cand in candidates:
            # 후보 위성 SGP4 객체 생성
            sat_o = Satrec.twoline2rv(cand["cand_sat"][1], cand["cand_sat"][2])
            n_steps = 600  # 10분간 1초 간격 샘플링
            # 기준 시각(ref_time, cand_time 중 더 이른 시각)에서 1분 전부터 샘플링
            times = min(cand["ref_time"], cand["cand_time"]) - timedelta(minutes=1)
            times = [times + timedelta(seconds=i * dt_s) for i in range(n_steps)]  # 샘플링 시각 리스트 생성

            min_dist = np.inf
            t_min = None
            min_s1, min_s2 = [], []
            ephemeris_sat1 = []
            ephemeris_sat2 = []
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
                # 기준 위성 위치 계산
                e1, r1, v1 = sat_ref.sgp4(jd, fr)
                # 후보 위성 위치 계산
                e2, r2, v2 = sat_o.sgp4(jd, fr)
                # SGP4 에러 발생 시 해당 시각 건너뜀
                if e1 != 0 or e2 != 0:
                    continue                

                ephemeris_sat1.append(np.hstack((t, r1, v1)))
                ephemeris_sat2.append(np.hstack((t, r2, v2)))

                r1 = np.array(r1)  # 기준 위성 위치 벡터
                r2 = np.array(r2)  # 후보 위성 위치 벡터
                d = np.linalg.norm(r1 - r2)  # 두 위성 간 거리 계산
                if d < min_dist:  # 최소 거리 및 해당 시각 갱신
                    min_dist = d
                    t_min = t
                    min_s1 = np.hstack((r1, v1))
                    min_s2 = np.hstack((r2, v2))

            # 최소 거리 조건(예: 10km 이하) 만족 시 결과에 추가
            if t_min is not None and min_dist < self.critera['minium_distance']:
                prob, rel = orcal.alfano_2d_collision_probability(min_s1, min_s2)
                results.append(
                    {
                        # 'type': cand['type'],
                        "sat1_norad": ref_sat[1][2:7].strip(),  # 기준 위성 NORAD 번호 추출
                        "sat2_norad": cand["cand_sat"][1][2:7].strip(),  # 상대 위성 NORAD 번호 추출
                        "ref_time": cand["ref_time"],  # 기준 위성 기준 시각
                        "other_time": cand["cand_time"],  # 상대 위성 기준 시각
                        "closest_distance_km": min_dist,  # 최소 접근 거리
                        "closest_time": t_min,  # 최소 접근 거리 발생 시각
                        "sat1_ephem": ephemeris_sat1,
                        "sat2_ephem": ephemeris_sat2,
                        "probability": prob,
                        "rel_vec": rel
                    }
                )
        return results

    def get_state_at_altitude(self, sat, start_time, target_altitude_km, max_days=30, step_minutes=10):
        """
        Propagate the satellite using SGP4 until it reaches the target altitude.
        Returns (reentry_time, state_vector) where:
        - reentry_time: datetime when altitude <= target_altitude_km
        - state_vector: (x, y, z, vx, vy, vz) in km and km/s
        """
        sat = Satrec.twoline2rv(sat[1], sat[2])
        ref_epoch = start_time
        for t in range(int((max_days * 24 * 60) / step_minutes)):
            jd, fr = jday(ref_epoch.year, ref_epoch.month, ref_epoch.day, ref_epoch.hour, ref_epoch.minute, ref_epoch.second + ref_epoch.microsecond / 1e6)
            e, r, v = sat.sgp4(jd, fr)
            # SGP4 에러 처리
            if e == 6:    # 위성 추락
                return 2, None, None
            elif e != 0:  # 기타 오류
                return 1, None, None
            alt = (r[0]**2 + r[1]**2 + r[2]**2)**0.5 - const.EARTH_RADIUS_KM
            if alt <= target_altitude_km:   # 정상적으로 목표 고도 도달
                return 0, ref_epoch, (r[0], r[1], r[2], v[0], v[1], v[2])
            ref_epoch += timedelta(minutes=step_minutes)
        # 목표 고도 도달 못함(오류)
        return 1, ref_epoch + timedelta(minutes=t), None



from MISC.DBmanager import DBmanager
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
