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

    @staticmethod
    def _batch_apogee_perigee(line2_list):
        """
        [최적화 2026-05-23] TLE line2 리스트에서 apogee/perigee를 numpy로 일괄 계산.
        기존: 위성마다 extract_elements + compute_apogee_perigee 호출 (Python 루프)
        개선: 문자열 슬라이싱 → numpy 배열 → 벡터 연산 (단 1회)

        반환: (apogee_arr, perigee_arr) — shape (N,), 단위 km (고도)
        """
        n_sats = len(line2_list)
        e_arr = np.empty(n_sats)
        n_arr = np.empty(n_sats)
        for k, l2 in enumerate(line2_list):
            e_arr[k] = float("0." + l2[26:33].strip())
            n_arr[k] = float(l2[52:63].strip())
        n_rad = n_arr * (2.0 * np.pi / 86400.0)
        a_arr = (const.MU / n_rad ** 2) ** (1.0 / 3.0)
        apogee  = a_arr * (1.0 + e_arr) - const.EARTH_RADIUS_KM
        perigee = a_arr * (1.0 - e_arr) - const.EARTH_RADIUS_KM
        return apogee, perigee

    def filter_altitude(self, tle_data, ref_line2, pad=None):
        """
        1단계: 고도 사전필터 (Apogee/Perigee Pre-filter)

        기준 위성의 고도 범위(근지점~원지점)와 겹치지 않는 위성을 제거한다.
        pad(여유값)는 adaptive: 궤도 장반경(a)의 2%, 최소 100km.

        [원리]
        - 두 위성의 고도 범위가 겹치지 않으면 물리적으로 충돌 불가능
        - 가장 빠르고 저비용인 필터 → 대부분의 위성을 여기서 제거

        [최적화 2026-05-23]
        - 기존: 위성마다 compute_apogee_perigee 개별 호출 (Python 루프 N회)
        - 개선: _batch_apogee_perigee로 전체 numpy 벡터 연산 (1회)
        - 정확도 변화 없음 (동일 공식)

        Args:
            tle_data: 전체 TLE 리스트 [(norad, line1, line2, epoch), ...]
            ref_line2: 기준 위성의 TLE line2
            pad: 고도 여유값(km). None이면 adaptive 계산
        """
        ref_apogee, ref_perigee = self.tle_manager.compute_apogee_perigee(ref_line2)
        ref_orb = self.tle_manager.extract_elements(ref_line2)
        adaptive_pad = max(100.0, 0.02 * ref_orb["a"])
        pad_km = float(adaptive_pad if pad is None else pad)

        # 전체 후보 위성 apogee/perigee를 numpy로 일괄 계산
        line2_list = [row[2] for row in tle_data]
        sat_apogee, sat_perigee = self._batch_apogee_perigee(line2_list)

        # 겹침 조건: sat_apogee >= ref_perigee - pad AND sat_perigee <= ref_apogee + pad
        keep = ~(
            (sat_apogee < ref_perigee - pad_km) | (sat_perigee > ref_apogee + pad_km)
        )
        return [tle_data[i] for i in np.where(keep)[0]]

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
    @staticmethod
    def _make_jd_arrays(t0: datetime, n_steps: int, dt_s: float):
        """
        [최적화 2026-05-23] Python 루프 jday 호출 → numpy 벡터 연산으로 대체.
        JD 기준점(t0)에 초 단위 오프셋을 더해 전체 배열을 한 번에 계산.
        반환: (jds, frs) — shape (n_steps,), 각각 정수부/소수부
        """
        jd0, fr0 = jday(t0.year, t0.month, t0.day,
                        t0.hour, t0.minute,
                        t0.second + t0.microsecond / 1e6)
        jd_total = (jd0 + fr0) + np.arange(n_steps) * (dt_s / 86400.0)
        jds = np.floor(jd_total)
        frs = jd_total - jds
        return jds, frs

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
        start = start_time if start_time is not None else ref_epoch
        end = end_time if end_time is not None else start + timedelta(days=analysis_days)

        sat_ref = self.satrec_from_tle(ref_l1, ref_l2)
        r1_start, v1_start = self.sgp4_rv_at(sat_ref, start)
        el1 = self.orcal.elements_from_rv(r1_start, v1_start)

        # ===== [최적화 2026-05-23] numpy 벡터화로 coarse 시간 배열 생성 =====
        base_coarse_dt = float(min_dt)
        total_seconds = (end - start).total_seconds()
        n_coarse = int(total_seconds / base_coarse_dt) + 1

        jds_coarse, frs_coarse = self._make_jd_arrays(start, n_coarse, base_coarse_dt)

        # 기준 위성 전파 (전체 그리드, 1회)
        ref_arr = SatrecArray([sat_ref])
        e_ref, r_ref, _ = ref_arr.sgp4(jds_coarse, frs_coarse)
        ref_pos = r_ref[0]   # (n_coarse, 3)
        ref_errs = e_ref[0]  # (n_coarse,)

        # ===== [최적화 2026-05-23] 후보 위성 전체를 단일 SatrecArray로 배치 전파 =====
        # 기존: 위성 1개씩 SatrecArray 생성 → N번 sgp4 호출
        # 개선: 전체 후보를 하나의 배열로 묶어 → 1번 sgp4 호출
        cand_list = []   # (other_sat, satrec, el2) 순서 보존
        for other_sat in tle_data:
            other_norad, o_l1, o_l2, _ = other_sat
            if other_norad == ref_sat[0]:
                continue
            sat_o = self.satrec_from_tle(o_l1, o_l2)
            try:
                r2_start, v2_start = self.sgp4_rv_at(sat_o, start)
            except RuntimeError:
                continue
            el2 = self.orcal.elements_from_rv(r2_start, v2_start)
            cand_list.append((other_sat, sat_o, el2))

        if not cand_list:
            return candidates

        # 전체 후보 위성 한 번에 coarse 전파
        batch_arr = SatrecArray([sr for _, sr, _ in cand_list])
        e_batch, r_batch, _ = batch_arr.sgp4(jds_coarse, frs_coarse)
        # e_batch: (n_cands, n_coarse), r_batch: (n_cands, n_coarse, 3)

        for ci, (other_sat, sat_o, el2) in enumerate(cand_list):
            other_norad = other_sat[0]
            o_pos  = r_batch[ci]   # (n_coarse, 3)
            o_errs = e_batch[ci]   # (n_coarse,)

            # ---- 쌍별 파라미터 ----
            if d_tol_km is None:
                pair_d_tol = max(25.0, min(100.0, 0.005 * min(el1["a"], el2["a"])))
            else:
                pair_d_tol = float(d_tol_km)

            v1_est = np.sqrt(const.MU / el1["a"])
            v2_est = np.sqrt(const.MU / el2["a"])
            v_rel_est = v1_est + v2_est

            if time_window is None:
                pair_time_window = max(600.0, min(3600.0, (pair_d_tol / v_rel_est) * 4.0))
            else:
                pair_time_window = float(time_window)

            # ---- 거리 배열 (벡터화) ----
            valid = (ref_errs == 0) & (o_errs == 0)
            dist_all = np.full(n_coarse, np.inf)
            dist_all[valid] = np.linalg.norm(ref_pos[valid] - o_pos[valid], axis=1)

            # ---- 극소점 + 임계값 이하 trigger ----
            coarse_dt_pair = self.choose_coarse_dt_by_period_and_speed(
                el1["a"], el2["a"], d_tol_km=pair_d_tol, min_dt=float(min_dt)
            )
            refine_trigger_km = max(pair_d_tol * 3.0, 0.5 * v_rel_est * coarse_dt_pair)
            pair_refine_step = max(1.0, min(float(refine_step_s),
                                            pair_d_tol / max(v_rel_est, 0.01) / 4.0))

            trigger_mask = np.zeros(n_coarse, dtype=bool)
            if n_coarse >= 3:
                trigger_mask[1:-1] |= (
                    (dist_all[1:-1] <= dist_all[:-2]) &
                    (dist_all[1:-1] <= dist_all[2:]) &
                    (dist_all[1:-1] <= refine_trigger_km)
                )
            trigger_mask |= (dist_all <= pair_d_tol)

            trigger_indices = np.where(trigger_mask)[0]
            if len(trigger_indices) == 0:
                continue

            # ---- trigger 인덱스 그룹화 ----
            last_added_t = None
            last_added_d = None
            groups = []
            current_group = [trigger_indices[0]]
            for k in range(1, len(trigger_indices)):
                if trigger_indices[k] - trigger_indices[k - 1] <= 2:
                    current_group.append(trigger_indices[k])
                else:
                    groups.append(current_group)
                    current_group = [trigger_indices[k]]
            groups.append(current_group)

            # ---- [최적화 2026-05-23] refinement 시간 배열도 numpy 벡터화 ----
            # SatrecArray는 그룹별로 ref/cand 2개짜리로 한 번에 전파
            refine_arr = SatrecArray([sat_ref, sat_o])

            for group in groups:
                t_center_idx = group[len(group) // 2]
                # datetime 없이 offset(초)으로 t_center 계산
                t_center_offset_s = t_center_idx * base_coarse_dt
                refine_start_offset = t_center_offset_s - coarse_dt_pair
                refine_total = 2.0 * coarse_dt_pair
                n_refine = int(refine_total / pair_refine_step) + 1

                # refine 구간 기준점: start + refine_start_offset
                refine_t0 = start + timedelta(seconds=refine_start_offset)
                jds_ref, frs_ref = self._make_jd_arrays(refine_t0, n_refine, pair_refine_step)

                e_rr, r_rr, _ = refine_arr.sgp4(jds_ref, frs_ref)
                # e_rr: (2, n_refine), r_rr: (2, n_refine, 3)
                valid_r = (e_rr[0] == 0) & (e_rr[1] == 0)
                if not np.any(valid_r):
                    continue

                d_refine = np.full(n_refine, np.inf)
                d_refine[valid_r] = np.linalg.norm(
                    r_rr[0][valid_r] - r_rr[1][valid_r], axis=1
                )
                min_idx = int(np.argmin(d_refine))
                best_d = d_refine[min_idx]
                if best_d > pair_d_tol:
                    continue

                best_t = refine_t0 + timedelta(seconds=min_idx * pair_refine_step)

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

        [최적화 2026-05-23]
        - 기존: Python 루프로 1초마다 개별 SGP4 호출 (600회/후보)
        - 개선: SatrecArray 2개짜리 배열로 전체 시간 배열 한 번에 전파
        - secondary refinement (0.1s)도 동일 방식으로 배치 처리
        - 정확도 변화 없음 (동일 SGP4 엔진, 동일 시간 그리드)

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
        sat_ref = Satrec.twoline2rv(ref_sat[1], ref_sat[2])

        for cand in candidates:
            sat_o = Satrec.twoline2rv(cand["cand_sat"][1], cand["cand_sat"][2])
            center = cand.get("t_center") or min(cand["ref_time"], cand["cand_time"])
            window_s = max(600.0, float(cand.get("time_window", 600.0)))
            n_steps = int(window_s / dt_s) + 1
            start_t = center - timedelta(seconds=window_s / 2)

            # [최적화 2026-05-23] numpy 벡터화로 JD 배열 생성 후 SatrecArray 배치 전파
            jds, frs = self._make_jd_arrays(start_t, n_steps, dt_s)
            pair_arr = SatrecArray([sat_ref, sat_o])
            e_pair, r_pair, v_pair = pair_arr.sgp4(jds, frs)
            # e_pair: (2, n_steps), r_pair: (2, n_steps, 3), v_pair: (2, n_steps, 3)

            valid = (e_pair[0] == 0) & (e_pair[1] == 0)
            if not np.any(valid):
                continue

            # 거리 배열 계산
            diff = r_pair[0] - r_pair[1]           # (n_steps, 3)
            dists = np.full(n_steps, np.inf)
            dists[valid] = np.linalg.norm(diff[valid], axis=1)

            min_idx = int(np.argmin(dists))
            min_dist = dists[min_idx]
            t_min = start_t + timedelta(seconds=min_idx * dt_s)
            min_s1 = np.hstack((r_pair[0][min_idx], v_pair[0][min_idx]))
            min_s2 = np.hstack((r_pair[1][min_idx], v_pair[1][min_idx]))

            # ephemeris 저장 (유효 시점만)
            valid_indices = np.where(valid)[0]
            ephemeris_sat1 = [
                np.hstack((start_t + timedelta(seconds=int(i) * dt_s),
                           r_pair[0][i], v_pair[0][i]))
                for i in valid_indices
            ]
            ephemeris_sat2 = [
                np.hstack((start_t + timedelta(seconds=int(i) * dt_s),
                           r_pair[1][i], v_pair[1][i]))
                for i in valid_indices
            ]

            # Secondary refinement: ±2s / 0.1s — 빠른 교차 포착
            if min_dist < self.critera['minium_distance'] * 3.0:
                refine_half = 2.0
                refine_dt = 0.1
                n_refine = int(2 * refine_half / refine_dt) + 1
                refine_t0 = t_min - timedelta(seconds=refine_half)
                jds_r, frs_r = self._make_jd_arrays(refine_t0, n_refine, refine_dt)
                e_r, r_r, v_r = pair_arr.sgp4(jds_r, frs_r)
                valid_r = (e_r[0] == 0) & (e_r[1] == 0)
                if np.any(valid_r):
                    d_r = np.full(n_refine, np.inf)
                    d_r[valid_r] = np.linalg.norm(r_r[0][valid_r] - r_r[1][valid_r], axis=1)
                    ri_min = int(np.argmin(d_r))
                    if d_r[ri_min] < min_dist:
                        min_dist = d_r[ri_min]
                        t_min = refine_t0 + timedelta(seconds=ri_min * refine_dt)
                        min_s1 = np.hstack((r_r[0][ri_min], v_r[0][ri_min]))
                        min_s2 = np.hstack((r_r[1][ri_min], v_r[1][ri_min]))

            if min_dist < self.critera['minium_distance']:
                prob, rel = orcal.alfano_2d_collision_probability(min_s1, min_s2)
                results.append({
                    "sat1_norad": ref_sat[0],
                    "sat2_norad": cand["cand_sat"][0],
                    "ref_time": cand["ref_time"],
                    "other_time": cand["cand_time"],
                    "closest_distance_km": min_dist,
                    "closest_time": t_min,
                    "sat1_ephem": ephemeris_sat1,
                    "sat2_ephem": ephemeris_sat2,
                    "probability": prob,
                    "rel_vec": rel,
                })
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
