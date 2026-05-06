"""
processor.py — Primary 위성별 CA 분석 워커 모듈

[역할]
  단일 primary 위성에 대한 전체 CA 파이프라인을 실행한다.
  ProcessPoolExecutor에 의해 병렬 호출되며, 각 워커는 독립적인
  DB 연결과 CA_filter 인스턴스를 생성한다.

[처리 흐름]
  1) 고도 필터 (filter_altitude) → 고도 범위 비교
  2) 궤도 경로 필터 (filter_orbitpath) → KDTree 3D 경로 비교
  3) 시간 필터 (filter_time) → SGP4 배치 전파로 근접 후보 추출
  4) 정밀 필터 (fine_filter_min_distance) → 1s+0.1s 샘플링
  5) 결과 구조화 → SAT_struc, COLLI_Info dict 생성

[입력]
  ref_sat: 기준 위성 TLE (norad, line1, line2, epoch)
  filtered_tle_snapshot: 전체 유효 TLE 리스트
  analysis_start / analysis_end: 분석 기간 (datetime, UTC)

[출력]
  COLLI dict 리스트 — 각 이벤트에 SAT_1, SAT_2, TCA, MIN_RNG,
  probability, Creation_date, COLLI_Info 포함
"""

from MISC.DBmanager import DBmanager
from CA.CA_filter import CA_filter
import MISC.Structurer as structer
from datetime import datetime, timezone
import time


def _safe_satcat_info(db_manager, norad):
    """SATCAT에서 위성 정보 조회 — 없으면 기본값 반환"""
    info = db_manager.get_SATCAT_info(norad)
    if info is None:
        return {
            "OBJECT_NAME": str(norad),
            "OBJECT_TYPE": "UNKNOWN",
            "RCS": 0,
        }
    return info


def process_primary(ref_sat, filtered_tle_snapshot, analysis_start, analysis_end, min_dt=120.0, refine_step_s=30.0):
    """
    단일 primary 위성에 대한 CA 파이프라인 워커 함수.
    ProcessPoolExecutor에서 병렬 호출됨.

    [처리 순서]
      1) filter_altitude   — 고도 범위 겹침 여부 확인
      2) filter_orbitpath   — 궤도 경로 3D 최소거리 확인 (KDTree)
      3) filter_time        — SGP4 배치 전파로 시간적 근접 후보 추출
      4) fine_filter_min_distance — 1s+0.1s 정밀 최소거리 탐색
      5) 결과 구조화         — SATCAT 정보 조회, SAT_struc/COLLI_Info 생성

    Args:
        ref_sat:     기준 위성 TLE (norad, line1, line2, epoch)
        filtered_tle_snapshot: 전체 유효 TLE 리스트
        analysis_start: 분석 시작 시각 (datetime, UTC)
        analysis_end:   분석 종료 시각 (datetime, UTC)
        min_dt:      시간필터 coarse step (초, 기본 120)
        refine_step_s: 시간필터 정밀 step (초, 기본 30)

    Returns:
        COLLI dict 리스트 (SAT_1, SAT_2, TCA, MIN_RNG, probability 등)
    """
    db_manager = DBmanager()       # 각 워커 프로세스에서 독립 DB 연결
    ca_filter = CA_filter()        # 필터 인스턴스 생성

    ref_line2 = ref_sat[2]

    # ── 1단계: 고도 필터 (원지점/근지점 겹침 확인) ──
    remain_tle = ca_filter.filter_altitude(filtered_tle_snapshot, ref_line2)

    # ── 2단계: 궤도 경로 필터 (KDTree + segment 최소거리) ──
    # 궤도 평면 각도 사전필터는 비활성 (교차 궤도면 접근 누락 방지)
    remain_tle = ca_filter.filter_orbitpath(remain_tle, ref_line2)

    # ── 3단계: 시간 필터 (SGP4 배치 전파, coarse→refine) ──
    remain_events = ca_filter.filter_time(
        ref_sat,
        remain_tle,
        analysis_days=1,
        time_window=None,
        d_tol_km=None,
        start_time=analysis_start,
        end_time=analysis_end,
        min_dt=min_dt,
        refine_step_s=refine_step_s,
    )

    # ── 4단계: 정밀 필터 (1s + 0.1s 샘플링으로 최소거리 확정) ──
    ca_res = ca_filter.fine_filter_min_distance(ref_sat, remain_tle, remain_events, dt_s=1.0)

    # ── 5단계: 결과 구조화 (SATCAT 정보 + CDM 형식 dict 생성) ──
    results = []
    for r in ca_res:
        orbit1 = structer.reassemble_orbit(r["sat1_ephem"])
        orbit2 = structer.reassemble_orbit(r["sat2_ephem"])
        sat1_info = _safe_satcat_info(db_manager, r["sat1_norad"])
        sat2_info = _safe_satcat_info(db_manager, r["sat2_norad"])
        SAT_1 = structer.SAT_struc(
            r["sat1_norad"], sat1_info["OBJECT_NAME"], sat1_info["OBJECT_TYPE"], 0, orbit1, 0, sat1_info["RCS"]
        )
        SAT_2 = structer.SAT_struc(
            r["sat2_norad"], sat2_info["OBJECT_NAME"], sat2_info["OBJECT_TYPE"], 0, orbit2, 0, sat2_info["RCS"]
        )
        COLLI = {
            "CDM_ID": f"{r['sat1_norad']}_{r['sat2_norad']}_{r['closest_time']}",
            # Creation date in UTC
            "Creation_date": datetime.now(timezone.utc),
            "TCA": r["closest_time"],
            "MIN_RNG": r["closest_distance_km"],
            "probability": r["probability"],
            "SAT_1": SAT_1,
            "SAT_2": SAT_2,
            "COLLI_Info": structer.COLLI_Info(r["rel_vec"][0:3], r["rel_vec"][3:6]),
        }
        results.append(COLLI)
    return results


def process_primary_profiled(ref_sat, filtered_tle_snapshot, analysis_start, analysis_end, min_dt=120.0, refine_step_s=30.0):
    """
    프로파일링 버전 — 각 단계별 소요시간과 후보 수를 로그로 출력.
    --profile-one 옵션 사용 시 호출됨.
    """
    db_manager = DBmanager()
    ca_filter = CA_filter()

    ref_line2 = ref_sat[2]

    t0 = time.perf_counter()
    remain_tle = ca_filter.filter_altitude(filtered_tle_snapshot, ref_line2)  # 1단계: 고도
    t1 = time.perf_counter()

    remain_tle = ca_filter.filter_orbitpath(remain_tle, ref_line2)  # 2단계: 궤도 경로
    t2 = time.perf_counter()

    remain_events = ca_filter.filter_time(  # 3단계: 시간 (배치 SGP4)
        ref_sat,
        remain_tle,
        analysis_days=1,
        time_window=None,
        d_tol_km=None,
        start_time=analysis_start,
        end_time=analysis_end,
        min_dt=min_dt,
        refine_step_s=refine_step_s,
    )
    t3 = time.perf_counter()

    ca_res = ca_filter.fine_filter_min_distance(ref_sat, remain_tle, remain_events, dt_s=1.0)  # 4단계: 정밀
    t4 = time.perf_counter()

    print(
        "프로파일 (초): "
        f"고도={t1 - t0:.2f}, 궤도경로={t2 - t1:.2f}, "
        f"시간={t3 - t2:.2f}, 정밀={t4 - t3:.2f} | "
        f"남은TLE={len(remain_tle)}, 후보={len(remain_events)}, 이벤트={len(ca_res)}"
    )

    results = []
    for r in ca_res:
        orbit1 = structer.reassemble_orbit(r["sat1_ephem"])
        orbit2 = structer.reassemble_orbit(r["sat2_ephem"])
        sat1_info = _safe_satcat_info(db_manager, r["sat1_norad"])
        sat2_info = _safe_satcat_info(db_manager, r["sat2_norad"])
        SAT_1 = structer.SAT_struc(
            r["sat1_norad"], sat1_info["OBJECT_NAME"], sat1_info["OBJECT_TYPE"], 0, orbit1, 0, sat1_info["RCS"]
        )
        SAT_2 = structer.SAT_struc(
            r["sat2_norad"], sat2_info["OBJECT_NAME"], sat2_info["OBJECT_TYPE"], 0, orbit2, 0, sat2_info["RCS"]
        )
        COLLI = {
            "CDM_ID": f"{r['sat1_norad']}_{r['sat2_norad']}_{r['closest_time']}",
            "Creation_date": datetime.now(timezone.utc),
            "TCA": r["closest_time"],
            "MIN_RNG": r["closest_distance_km"],
            "probability": r["probability"],
            "SAT_1": SAT_1,
            "SAT_2": SAT_2,
            "COLLI_Info": structer.COLLI_Info(r["rel_vec"][0:3], r["rel_vec"][3:6]),
        }
        results.append(COLLI)
    return results
