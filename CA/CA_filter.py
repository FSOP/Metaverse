"""
CA_filter.py — 충돌 분석(Conjunction Assessment) 핵심 필터링 모듈

[역할]
  위성 충돌 위험 분석의 핵심 단계별 필터링을 수행한다.
  전체 TLE 카탈로그(수천~수만 개)에서 기준 위성과 근접할 가능성이 있는
  후보를 단계적으로 줄여나가는 "깔때기(funnel)" 구조이다.

[필터 단계]
  1) filter_altitude    — 고도(원지점/근지점) 사전필터
  2) filter_orbitpath   — 궤도 경로 3D 거리 사전필터 (KDTree)
  3) filter_time        — SGP4 배치 전파 시간필터 (coarse→refine)
  4) fine_filter_min_distance — 정밀 최소거리 탐색 (1s + 0.1s)

[주요 알고리즘]
  - SatrecArray 배치 SGP4: 수천 위성을 벡터화하여 한 번에 전파
  - KDTree + segment-segment 최소거리: 궤도 경로 비교 가속화
  - Alfano 2D 최대 충돌확률: 만남 평면(encounter plane) 투영 후 계산
"""

from datetime import datetime, timedelta
from MISC.TLEmanager import TLEmanager
from CA.propagators import propagators
from CA.orbitcalculator import orcal
from sgp4.api import Satrec, SatrecArray, jday
from math import acos, atan2, sqrt, pi
try:
    from scipy.spatial import cKDTree as KDTree
except Exception:
    KDTree = None

import MISC.constants as const
import numpy as np


class CA_filter:
    def __init__(self):
        self.tle_manager = TLEmanager()
        self.propagator = propagators()
        self.orcal = orcal()
        # 최소 접근 거리 임계값 (km) — SOCRATES 기준 5km
        self.critera = {
            'minium_distance': 5.0,  # km — SOCRATES 5 km max range 기준
        }

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

    def filter_altitude(self, tle_data, ref_line2, pad=None):
        """
        1단계: 고도 사전필터 (Apogee/Perigee Pre-filter)

        기준 위성의 고도 범위(근지점~원지점)와 겹치지 않는 위성을 제거한다.
        pad(여유값)는 adaptive: 궤도 장반경(a)의 2%, 최소 100km.

        [원리]
        - 두 위성의 고도 범위가 겹치지 않으면 물리적으로 충돌 불가능
        - 가장 빠르고 저비용인 필터 → 대부분의 위성을 여기서 제거

        Args:
            tle_data: 전체 TLE 리스트 [(norad, line1, line2, epoch), ...]
            ref_line2: 기준 위성의 TLE line2
            pad: 고도 여유값(km). None이면 adaptive 계산

        """
        filtered_tles = []
        ref_apogee, ref_perigee = self.tle_manager.compute_apogee_perigee(ref_line2)
        ref_orb = self.tle_manager.extract_elements(ref_line2)
        # Adaptive pad: 2% of semi-major axis, with a minimum of 100 km,
        # unless an explicit pad is provided.
        adaptive_pad = max(100.0, 0.02 * ref_orb["a"])
        pad_km = float(adaptive_pad if pad is None else pad)

        for i in range(len(tle_data)):
            norad, line1, line2, tle_epoch = tle_data[i]
            sat2_apogee, sat2_perigee = self.tle_manager.compute_apogee_perigee(line2)

            # Check if the apogee and perigee are within the specified pad
            if not (
                (sat2_apogee < ref_perigee - pad_km) or (sat2_perigee > ref_apogee + pad_km)
            ):
                filtered_tles.append((norad, line1, line2, tle_epoch))

        return filtered_tles

    def filter_orbitpath(self, tle_data, ref_line2, N_points=None, threshold=300):
        """
        2단계: 궤도 경로 사전필터 (Orbit Path Pre-filter)

        기준 위성과 후보 위성의 Keplerian 궤도 경로를 3D 공간에서 비교하여,
        경로간 최소거리가 threshold(300km) 이상인 위성을 제거한다.

        [원리]
        - 궤도 경로(orbit path)는 케플러 궤도의 공간상 형태를 의미
        - arc-length 기반 균등 샘플링으로 궤도를 3D 점으로 표현
        - KDTree 반경 질의로 빠르게 근접 여부 판단
        - 근접 후보 발견 시 segment-segment 최소거리로 정밀 확인

        [성능]
        - KDTree 덕분에 N×M 브루트포스 대비 O(N log N) 수준
        - 고도 필터 통과한 수백~수천 위성 → 수십~수백으로 축소

        Args:
            tle_data: 고도 필터를 통과한 TLE 리스트
            ref_line2: 기준 위성의 TLE line2
            N_points: 궤도 샘플링 점 수 (None이면 adaptive)
            threshold: 경로 최소거리 임계값 (km, 기본 300km)

        Returns:
            궤도 경로가 threshold 이내로 접근하는 TLE 리스트
        """
        filtered_tles = []
        # Generate a reference orbit path using the reference satellite's TLE
        ref_orb = self.tle_manager.extract_elements(ref_line2)

        # plane-angle prefilter removed to avoid false negatives on cross-plane encounters.
        # Distance-based and KDTree segment checks remain.

        # Adaptive sampling: target arc length per sample (km)
        max_points = 720
        if N_points is None:
            target_arc_km = 200.0
            min_points = 36
            n_est = int(np.ceil((2 * np.pi * ref_orb["a"]) / target_arc_km))
            N_points = max(min_points, min(max_points, n_est))
        else:
            N_points = int(N_points)

        # Use arc-length based theta sampling for reference orbit
        theta_ref = self._arc_length_samples(ref_orb, N_points)
        ref_r = self.propagator.orbit_path(ref_orb, theta_ref)  # Reference orbit path (3, N_points)

        # Sampling-based path comparison can miss close approaches if threshold is too small
        # relative to the arc length between samples. Use an adaptive minimum threshold to
        # avoid false negatives from coarse sampling.
        # Arc length per sample (km) ≈ 2πa / N_points
        arc_step_km = (2 * np.pi * ref_orb["a"]) / max(1, int(N_points))
        effective_threshold = max(float(threshold), 0.5 * arc_step_km)

        # Prepare KDTree if available for faster neighbor queries
        ref_points = ref_r.T  # shape (N_points, 3)
        tree = KDTree(ref_points) if KDTree is not None else None

        # Iterate through the TLE except the reference TLE
        for i in range(len(tle_data)):
            norad, line1, line2, tle_epoch = tle_data[i]
            orbit = self.tle_manager.extract_elements(line2)

            # No plane-angle prefilter: proceed to sampling + KDTree/segment checks

            # Compare the orbit with the reference orbit (sampled) using arc-length samples per orbit
            theta_other = self._arc_length_samples(orbit, N_points)
            sat2_r = self.propagator.orbit_path(orbit, theta_other)
            sat_points = sat2_r.T  # shape (N_points, 3)

            flag_ca = False
            if tree is not None:
                # use KDTree radius queries for each sample point (vectorized)
                neighbors = tree.query_ball_point(sat_points, r=effective_threshold)
                # neighbors is a list of lists; refine using segment-segment checks
                all_empty = all((not nb) for nb in neighbors)
                # If KDTree returned no neighbors for sampled points, attempt a denser local sampling
                if all_empty:
                    N_dense = min(int(N_points * 5), max_points)
                    theta_other_dense = self._arc_length_samples(orbit, N_dense)
                    sat2_r_dense = self.propagator.orbit_path(orbit, theta_other_dense)
                    sat_points_dense = sat2_r_dense.T
                    neighbors_dense = tree.query_ball_point(sat_points_dense, r=effective_threshold)
                    # Switch to the dense sample points so neighbor indices align
                    sat_points = sat_points_dense
                    neighbors = neighbors_dense
                for j_idx, nb in enumerate(neighbors):
                    if not nb:
                        continue
                    # for each nearby reference sample index, check adjacent segments
                    for ref_idx in nb:
                        # build ref segment [ref_idx, ref_idx+1]
                        i0 = ref_idx
                        i1 = (ref_idx + 1) % ref_points.shape[0]
                        p1 = ref_points[i0]
                        p2 = ref_points[i1]
                        # build sat segment [j_idx, j_idx+1]
                        j0 = j_idx
                        j1 = (j_idx + 1) % sat_points.shape[0]
                        q1 = sat_points[j0]
                        q2 = sat_points[j1]
                        dseg = self._segment_segment_min_distance(p1, p2, q1, q2)
                        if dseg < effective_threshold:
                            flag_ca = True
                            break
                        # also check previous segments (ref_idx-1, ref_idx) and (j_idx-1, j_idx)
                        i_prev = (ref_idx - 1) % ref_points.shape[0]
                        j_prev = (j_idx - 1) % sat_points.shape[0]
                        p1b = ref_points[i_prev]
                        p2b = ref_points[ref_idx]
                        q1b = sat_points[j_prev]
                        q2b = sat_points[j_idx]
                        dseg2 = self._segment_segment_min_distance(p1b, p2b, q1b, q2b)
                        if dseg2 < effective_threshold:
                            flag_ca = True
                            break
                    if flag_ca:
                        break
            else:
                # fallback: brute-force distance matrix (original method)
                distance = np.linalg.norm(ref_r[:, :, None] - sat2_r[:, None, :], axis=0)
                flag_ca = np.any(distance < effective_threshold)

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

    def _segment_segment_min_distance(self, p1, p2, q1, q2):
        """Compute minimum distance between two line segments p(s)=p1+s*(p2-p1), s in [0,1]
        and q(t)=q1+t*(q2-q1), t in [0,1]. Uses algorithm from Ericson (Real-Time Collision Detection).
        Returns the scalar distance.
        """
        u = p2 - p1
        v = q2 - q1
        w = p1 - q1
        a = np.dot(u, u)  # always >= 0
        b = np.dot(u, v)
        c = np.dot(v, v)  # always >= 0
        d = np.dot(u, w)
        e = np.dot(v, w)
        D = a * c - b * b  # denominator
        sc = 0.0
        sN = 0.0
        sD = D
        tc = 0.0
        tN = 0.0
        tD = D

        SMALL_NUM = 1e-12

        if D < SMALL_NUM:  # lines almost parallel
            sN = 0.0
            sD = 1.0
            tN = e
            tD = c
        else:
            sN = (b * e - c * d)
            tN = (a * e - b * d)
            if sN < 0.0:
                sN = 0.0
                tN = e
                tD = c
            elif sN > sD:
                sN = sD
                tN = e + b
                tD = c

        if tN < 0.0:
            tN = 0.0
            if -d < 0.0:
                sN = 0.0
            elif -d > a:
                sN = sD
            else:
                sN = -d
                sD = a
        elif tN > tD:
            tN = tD
            if (-d + b) < 0.0:
                sN = 0
            elif (-d + b) > a:
                sN = sD
            else:
                sN = (-d + b)
                sD = a

        sc = 0.0 if abs(sN) < SMALL_NUM else sN / sD
        tc = 0.0 if abs(tN) < SMALL_NUM else tN / tD

        dP = w + (sc * u) - (tc * v)
        return np.linalg.norm(dP)

    def _arc_length_samples(self, orbit, N_points, oversample=8):
        """Return theta sample array (radians) approximately equally spaced by arc length along orbit.
        Uses an oversampled uniform theta grid and selects N_points by cumulative arc distance.
        """
        # create fine theta grid
        nf = max(int(oversample * N_points), 360)
        theta_fine = np.linspace(0, 2 * np.pi, nf, endpoint=False)
        r_fine = self.propagator.orbit_path(orbit, theta_fine)  # shape (3, nf)
        pts = r_fine.T  # (nf,3)
        # cumulative distance along orbit (closed loop)
        seg_d = np.linalg.norm(np.diff(np.vstack([pts, pts[0]]), axis=0), axis=1)
        cum = np.cumsum(seg_d)
        total = cum[-1]
        # target distances
        targets = np.linspace(0, total, N_points, endpoint=False)
        idxs = np.searchsorted(cum, targets)
        idxs = np.mod(idxs, nf)
        return theta_fine[idxs]

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

    # ─────────────────────────────────────────────────
    # 3단계: 시간 기반 사전필터 (Time Pre-filter)
    # ─────────────────────────────────────────────────
    # [원리]
    #   SGP4로 실제 위성 위치를 시간에 따라 전파하여 거리를 계산한다.
    #   SatrecArray 배치 API를 사용하여 전체 시간 그리드를 한 번에 전파하며,
    #   numpy 벡터 연산으로 거리 배열을 계산한다.
    #
    # [과정]
    #   1) 기준 위성의 위치를 coarse 그리드(120s)로 미리 계산 (전체 분석 기간)
    #   2) 각 후보 위성을 동일 그리드로 배치 전파 → 거리 배열 생성
    #   3) 거리 배열에서 극소점(local min) + 임계값 이하 구간 탐색
    #   4) 발견된 구간에서 adaptive refine step(1~30s)으로 정밀 탐색
    #   5) pair_d_tol(25~100km) 이내 접근 시점을 fine_filter 후보로 등록
    #
    # [성능]
    #   기존 per-satellite Python 루프 대비 4~5배 빠름
    #   (662s → 136s for NORAD 64586, 분석기간 7일)
    # ─────────────────────────────────────────────────
    def filter_time(
        self,
        ref_sat,
        tle_data,
        analysis_days=10,
        time_window=None,
        d_tol_km=None,
        start_time=None,
        end_time=None,
        min_dt=120.0,
        refine_step_s=30.0,
    ):
        _, ref_l1, ref_l2, ref_epoch = ref_sat

        candidates = []
        # 분석 시작/종료 시각 결정
        start = start_time if start_time is not None else ref_epoch
        end = end_time if end_time is not None else start + timedelta(days=analysis_days)

        # 기준 위성 SGP4 객체 생성
        sat_ref = self.satrec_from_tle(ref_l1, ref_l2)
        r1_start, v1_start = self.sgp4_rv_at(sat_ref, start)
        el1 = self.orcal.elements_from_rv(r1_start, v1_start)

        # ===== 기준 위성 coarse 그리드 사전 계산 (한 번만) =====
        # 기본 coarse_dt = min_dt (120s) 고정으로 reference 그리드 구축
        base_coarse_dt = float(min_dt)
        total_seconds = (end - start).total_seconds()
        n_coarse = int(total_seconds / base_coarse_dt) + 1

        # 시간 배열 생성 (JD, FR 형식)
        jds_coarse = np.empty(n_coarse)
        frs_coarse = np.empty(n_coarse)
        times_coarse = []  # datetime 객체 리스트 (나중에 필요)
        for i in range(n_coarse):
            t_i = start + timedelta(seconds=i * base_coarse_dt)
            times_coarse.append(t_i)
            jd_i, fr_i = jday(t_i.year, t_i.month, t_i.day,
                              t_i.hour, t_i.minute,
                              t_i.second + t_i.microsecond / 1e6)
            jds_coarse[i] = jd_i
            frs_coarse[i] = fr_i

        # 기준 위성 배치 전파 (전체 시간 그리드)
        ref_arr = SatrecArray([sat_ref])
        e_ref, r_ref, v_ref = ref_arr.sgp4(jds_coarse, frs_coarse)
        # r_ref shape: (1, n_coarse, 3) → (n_coarse, 3)
        ref_pos = r_ref[0]  # (n_coarse, 3)
        ref_errs = e_ref[0]  # (n_coarse,) — 0이면 정상

        # ===== 각 후보 위성에 대해 배치 SGP4로 거리 계산 =====
        for other_sat in tle_data:
            other_norad, o_l1, o_l2, o_epoch = other_sat
            # 자기 자신은 건너뜀
            if other_norad == ref_sat[0]:
                continue

            sat_o = self.satrec_from_tle(o_l1, o_l2)
            try:
                r2_start, v2_start = self.sgp4_rv_at(sat_o, start)
            except RuntimeError:
                continue
            el2 = self.orcal.elements_from_rv(r2_start, v2_start)

            # ---- 쌍별 파라미터 계산 ----
            if d_tol_km is None:
                base = 0.005 * min(el1["a"], el2["a"])
                pair_d_tol = max(25.0, min(100.0, base))
            else:
                pair_d_tol = float(d_tol_km)

            v1_est = np.sqrt(const.MU / el1["a"])
            v2_est = np.sqrt(const.MU / el2["a"])
            v_rel_est = v1_est + v2_est

            if time_window is None:
                pair_time_window = max(600.0, min(3600.0, (pair_d_tol / v_rel_est) * 4.0))
            else:
                pair_time_window = float(time_window)

            # ---- 후보 위성 배치 전파 ----
            o_arr = SatrecArray([sat_o])
            e_o, r_o, v_o = o_arr.sgp4(jds_coarse, frs_coarse)
            o_pos = r_o[0]  # (n_coarse, 3)
            o_errs = e_o[0]  # (n_coarse,)

            # 유효한 시점만 사용 (양쪽 모두 에러 없는 시점)
            valid = (ref_errs == 0) & (o_errs == 0)

            # 거리 배열 계산 (벡터화)
            diff = ref_pos - o_pos  # (n_coarse, 3)
            dist_all = np.full(n_coarse, np.inf)
            dist_all[valid] = np.linalg.norm(diff[valid], axis=1)

            # ---- 극소점 + 임계값 이하 시점 탐색 ----
            # refine_trigger: 이 거리 이하인 시점에서만 정밀 탐색 수행
            coarse_dt_pair = self.choose_coarse_dt_by_period_and_speed(
                el1["a"], el2["a"], d_tol_km=pair_d_tol, min_dt=float(min_dt)
            )
            refine_trigger_km = max(pair_d_tol * 3.0, 0.5 * v_rel_est * coarse_dt_pair)
            pair_refine_step = max(1.0, min(float(refine_step_s),
                                            pair_d_tol / max(v_rel_est, 0.01) / 4.0))

            # 극소점 탐색: dist[i-1] >= dist[i] <= dist[i+1] 이고 dist[i] < refine_trigger
            # 직접 임계값 이하 시점: dist[i] < pair_d_tol
            trigger_mask = np.zeros(n_coarse, dtype=bool)

            # 3-point 극소점
            if n_coarse >= 3:
                local_min = (
                    (dist_all[1:-1] <= dist_all[:-2]) &
                    (dist_all[1:-1] <= dist_all[2:]) &
                    (dist_all[1:-1] <= refine_trigger_km)
                )
                trigger_mask[1:-1] |= local_min

            # 직접 임계값 이하
            trigger_mask |= (dist_all <= pair_d_tol)

            trigger_indices = np.where(trigger_mask)[0]
            if len(trigger_indices) == 0:
                continue

            # ---- 정밀 refinement (trigger 시점 근처에서) ----
            last_added_t = None
            last_added_d = None

            # 연속된 trigger 인덱스를 그룹으로 묶어 한 번만 refine
            groups = []
            current_group = [trigger_indices[0]]
            for k in range(1, len(trigger_indices)):
                if trigger_indices[k] - trigger_indices[k-1] <= 2:
                    current_group.append(trigger_indices[k])
                else:
                    groups.append(current_group)
                    current_group = [trigger_indices[k]]
            groups.append(current_group)

            for group in groups:
                # 그룹의 시간 범위: 그룹 첫/끝 인덱스 ± coarse_dt_pair
                t_center_idx = group[len(group)//2]
                t_center = times_coarse[t_center_idx]

                # 정밀 refinement 범위
                refine_start = t_center - timedelta(seconds=coarse_dt_pair)
                refine_end = t_center + timedelta(seconds=coarse_dt_pair)

                # 정밀 시간 그리드 생성
                refine_total = (refine_end - refine_start).total_seconds()
                n_refine = int(refine_total / pair_refine_step) + 1
                jds_ref = np.empty(n_refine)
                frs_ref = np.empty(n_refine)
                times_ref = []
                for ri in range(n_refine):
                    rt = refine_start + timedelta(seconds=ri * pair_refine_step)
                    times_ref.append(rt)
                    jd_r, fr_r = jday(rt.year, rt.month, rt.day,
                                      rt.hour, rt.minute,
                                      rt.second + rt.microsecond / 1e6)
                    jds_ref[ri] = jd_r
                    frs_ref[ri] = fr_r

                # 배치 SGP4 — 기준위성과 후보위성 동시 정밀 전파
                e_rr, r_rr, _ = ref_arr.sgp4(jds_ref, frs_ref)
                e_or, r_or, _ = o_arr.sgp4(jds_ref, frs_ref)
                valid_r = (e_rr[0] == 0) & (e_or[0] == 0)

                if not np.any(valid_r):
                    continue

                d_refine = np.full(n_refine, np.inf)
                d_refine[valid_r] = np.linalg.norm(r_rr[0][valid_r] - r_or[0][valid_r], axis=1)

                min_idx = np.argmin(d_refine)
                best_d = d_refine[min_idx]
                best_t = times_ref[min_idx]

                if best_d > pair_d_tol:
                    continue

                # 중복 방지: 이전에 추가된 후보와의 시간 차이 확인
                new_cand = {
                    "ref_time": best_t,
                    "cand_time": best_t,
                    "t_center": best_t,
                    "time_window": pair_time_window,
                    "ref_sat": ref_sat,
                    "cand_sat": other_sat,
                }
                if last_added_t is None or abs((best_t - last_added_t).total_seconds()) > pair_time_window:
                    candidates.append(new_cand)
                    last_added_t = best_t
                    last_added_d = best_d
                else:
                    # 동일 시간대면 더 가까운 것만 유지
                    if best_d < last_added_d:
                        for idx_rev in range(len(candidates) - 1, -1, -1):
                            c = candidates[idx_rev]
                            if c.get("cand_sat") and c["cand_sat"][0] == other_norad:
                                candidates[idx_rev] = new_cand
                                last_added_t = best_t
                                last_added_d = best_d
                                break

        return candidates

    def fine_filter_min_distance(self, ref_sat, other_sats, candidates, dt_s=1.0):
        """
        4단계: 정밀 최소거리 탐색 (Fine Filter)

        시간필터를 통과한 후보 쌍에 대해, 정밀 SGP4 전파로 최소 접근 거리를 계산.

        [원리]
        - 각 후보의 TCA 부근 ±300s 구간에서 1초 간격으로 SGP4 전파
        - 최소 거리점 발견 후 ±2s 범위에서 0.1s 정밀 탐색 (secondary refinement)
        - 최종 최소거리 < 5km 이면 이벤트로 확정
        - Alfano 2D 최대 충돌확률 계산 (만남 평면 투영 기반)

        [성능]
        - 후보 수가 적으므로 (일반적으로 수십 개) 순차 처리해도 빠름

        Args:
            ref_sat:    기준 위성 TLE 튜플 (norad, line1, line2, epoch)
            other_sats: 비교 대상 위성 리스트 (미사용, candidates에 포함됨)
            candidates: filter_time을 통과한 후보 리스트 (dict)
            dt_s:       1차 샘플링 간격 (초, 기본 1.0)

        Returns:
            list of dict — 각 이벤트:
              sat1_norad, sat2_norad, closest_distance_km, closest_time,
              sat1_ephem, sat2_ephem, probability, rel_vec
        """
        results = []
        # 기준 위성 SGP4 객체 생성
        sat_ref = Satrec.twoline2rv(ref_sat[1], ref_sat[2])

        for cand in candidates:
            # 후보 위성 SGP4 객체 생성
            sat_o = Satrec.twoline2rv(cand["cand_sat"][1], cand["cand_sat"][2])
            # Candidate-centered sampling window (minimum 600s for adequate TCA search)
            center = cand.get("t_center") or min(cand["ref_time"], cand["cand_time"])
            window_s = max(600.0, float(cand.get("time_window", 600.0)))
            n_steps = int(window_s / dt_s) + 1
            start_t = center - timedelta(seconds=window_s / 2)
            times = [start_t + timedelta(seconds=i * dt_s) for i in range(n_steps)]

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

            # Secondary refinement: finer 0.1s sampling around detected minimum
            # to catch fast crossings where 1s sampling is too coarse
            if t_min is not None and min_dist < self.critera['minium_distance'] * 3.0:
                refine_half = 2.0  # ±2s around coarse minimum
                refine_dt = 0.1
                n_refine = int(2 * refine_half / refine_dt) + 1
                for ri in range(n_refine):
                    rt = t_min - timedelta(seconds=refine_half) + timedelta(seconds=ri * refine_dt)
                    jd_r, fr_r = jday(rt.year, rt.month, rt.day, rt.hour, rt.minute,
                                      rt.second + rt.microsecond / 1e6)
                    e1r, r1r, v1r = sat_ref.sgp4(jd_r, fr_r)
                    e2r, r2r, v2r = sat_o.sgp4(jd_r, fr_r)
                    if e1r != 0 or e2r != 0:
                        continue
                    dr = np.linalg.norm(np.array(r1r) - np.array(r2r))
                    if dr < min_dist:
                        min_dist = dr
                        t_min = rt
                        min_s1 = np.hstack((r1r, v1r))
                        min_s2 = np.hstack((r2r, v2r))

            # 최소 거리 조건 만족 시 결과에 추가
            if t_min is not None and min_dist < self.critera['minium_distance']:
                prob, rel = orcal.alfano_2d_collision_probability(min_s1, min_s2)
                results.append(
                    {
                        # 'type': cand['type'],
                        "sat1_norad": ref_sat[0],  # 기준 위성 NORAD 번호 (tuple element)
                        "sat2_norad": cand["cand_sat"][0],  # 상대 위성 NORAD 번호 (tuple element)
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



# module intended for import; example usage removed
