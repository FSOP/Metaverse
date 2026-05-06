"""
ConjunctionAssessment.py — 위성 충돌 위험 분석 (Conjunction Assessment) 메인 드라이버

[충돌 분석(CA) 파이프라인 개요 / Pipeline Overview]
──────────────────────────────────────────────────────────────
  1단계. TLE 로딩 및 필터링 — DB에서 최신 TLE(NORAD별 1건) 가져오기,
         분석 기간 ± pad 이내의 유효 TLE만 사용
  2단계. 고도 필터(Apogee/Perigee Pre-filter) — 기준 위성의 고도 범위와
         겹치지 않는 위성 제거 (adaptive pad: semi-major axis 2%, 최소 100km)
  3단계. 궤도 경로 필터(Orbit Path Pre-filter) — Keplerian 궤도 arc-length
         기반 균등 샘플링 + KDTree 최소거리 비교 (>300km 제거)
  4단계. 시간 필터(Time Pre-filter, SGP4 배치 전파) — 120s coarse 그리드 →
         극소점+임계값 구간 탐색 → adaptive refine(1~30s) → pair_d_tol 이내 등록
  5단계. 정밀 필터(Fine Filter) — ±300s에서 1s 간격 SGP4 → 0.1s 정밀 탐색 →
         최소거리 < 5km 확정 + Alfano 2D 최대 충돌확률 계산
  6단계. 중복 제거 + DB 저장 + API 전송 — 동일 쌍 300초 내 클러스터링,
         DB insert_CA, API /api/ca_events 전송
──────────────────────────────────────────────────────────────

[사용법 / Usage]
  python ConjunctionAssessment.py                       # 모든 primary 위성 분석
  python ConjunctionAssessment.py --norad 64586         # 특정 위성만 분석
  python ConjunctionAssessment.py --no-send             # API 전송 비활성화
  python ConjunctionAssessment.py --no-db               # DB 저장 비활성화
  python ConjunctionAssessment.py --profile-one 64586   # 프로파일링 모드
  python ConjunctionAssessment.py --fast                # 속도 우선 프리셋
"""

from MISC.DBmanager import DBmanager
from MISC.TLEmanager import TLEmanager
from CA.CA_filter import CA_filter
from CA.processor import process_primary
from MISC import config
from MISC.ca_api import CAEventSender
from datetime import datetime, timedelta, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
import logging
import argparse
import sys

# ─── 설정값 로드 / Load configuration ───
ANALYSIS_DURATION = config.ANALYSIS_DURATION   # 분석 기간 (일)
TLE_AGE_LIMIT = config.TLE_AGE_LIMIT           # TLE 유효기간 (일)
API_BASE_URL = config.API_BASE_URL             # API 서버 주소
BATCH_SIZE = config.BATCH_SIZE                 # 배치 전송 크기 (0=즉시전송)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("ca_api_errors.log"),
    ],
)


# ─── 중복 제거 함수 / Deduplication ───
def dedupe_events(events, window_s=300):
    """
    동일 위성 쌍의 근접 이벤트를 시간 윈도(window_s) 기준으로 클러스터링하여 중복 제거.
    각 클러스터에서 최소 거리 이벤트만 유지한다.

    Args:
        events:   CA 이벤트 dict 리스트 (SAT_1, SAT_2, TCA, MIN_RNG 등 포함)
        window_s: 중복 판정 시간 윈도 (초, 기본 300초 = 5분)

    Returns:
        중복 제거된 이벤트 리스트
    """
    # 1) 위성 쌍(pair) 기준으로 그룹핑 — 순서 무관 (min, max)
    groups = defaultdict(list)
    for e in events:
        sat1 = e.get("SAT_1", {}) if isinstance(e.get("SAT_1"), dict) else {}
        sat2 = e.get("SAT_2", {}) if isinstance(e.get("SAT_2"), dict) else {}
        s1 = int(sat1.get("SAT_ID", 0) or 0)
        s2 = int(sat2.get("SAT_ID", 0) or 0)
        key = (min(s1, s2), max(s1, s2)) if s1 and s2 else (s1, s2)
        groups[key].append(e)

    # 2) 각 그룹 내에서 TCA 순 정렬 후 클러스터링
    kept = []
    for key, lst in groups.items():
        lst_sorted = sorted(lst, key=lambda x: x.get("TCA") or x.get("tca"))
        cluster = []
        for ev in lst_sorted:
            t = ev.get("TCA") or ev.get("tca")
            if not cluster:
                cluster.append(ev)
                continue
            last_t = cluster[-1].get("TCA") or cluster[-1].get("tca")
            try:
                delta = abs((t - last_t).total_seconds())
            except Exception:
                delta = 0
            if delta <= window_s:
                cluster.append(ev)  # 같은 클러스터에 추가
            else:
                # 클러스터 종료 → 최소 거리 이벤트만 보존
                best = min(cluster, key=lambda x: float(x.get("MIN_RNG", 1e9)))
                kept.append(best)
                cluster = [ev]
        if cluster:
            best = min(cluster, key=lambda x: float(x.get("MIN_RNG", 1e9)))
            kept.append(best)
    return kept


# ──────────────────────────────────────────────────────────
#  메인 실행 블록 / Main execution
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='위성 충돌 위험 분석 (Conjunction Assessment)')
    parser.add_argument('--norad', '-n', type=int, help='특정 primary NORAD ID만 분석')
    parser.add_argument('--dryrun', action='store_true', help='데이터 준비만 하고 분석 실행 안 함')
    parser.add_argument('--no-send', action='store_true', help='API 서버 전송 비활성화')
    parser.add_argument('--no-db', action='store_true', help='DB 저장 비활성화')
    parser.add_argument('--analysis-days', type=float, default=None, help='분석 기간 (일) 오버라이드')
    parser.add_argument('--profile-one', type=int, default=None, help='단일 primary 프로파일링 모드')
    parser.add_argument('--min-dt', type=float, default=120.0, help='시간필터 coarse step (초)')
    parser.add_argument('--refine-step', type=float, default=30.0, help='시간필터 정밀 step (초)')
    parser.add_argument('--fast', action='store_true', help='속도 우선 프리셋 (min_dt=240, refine_step=60)')
    args = parser.parse_args()

    # ─── 속도 프리셋 적용 / Apply fast preset ───
    if args.fast:
        args.min_dt = max(float(args.min_dt), 240.0)
        args.refine_step = max(float(args.refine_step), 60.0)

    # ─── 데이터 로딩 / Load data ───
    tle_manager = TLEmanager()
    db_manager = DBmanager()

    count_tle = db_manager.get_tle_count()
    print(f"[데이터] DB 내 총 TLE 수: {count_tle}")

    primary_norad_list = db_manager.get_primary_norad_list()
    print(f"[데이터] 등록된 primary 위성 수: {len(primary_norad_list)}")

    now_epoch = config.now_epoch()  # 현재 UTC 시각
    analysis_start = now_epoch
    analysis_days = float(args.analysis_days) if args.analysis_days is not None else float(ANALYSIS_DURATION)
    analysis_end = analysis_start + timedelta(days=analysis_days)
    print(f"[설정] 분석 기간: {analysis_start:%Y-%m-%d %H:%M} → {analysis_end:%Y-%m-%d %H:%M} ({analysis_days}일)")

    # 분석 시점 기준 최신 TLE 스냅샷 (NORAD별 1건)
    tle_all = db_manager.get_latest_TLEs_asof(analysis_start)
    filtered_tle = tle_manager.filter_outdated_tles(
        tle_all, analysis_start, analysis_end, pad_days=TLE_AGE_LIMIT
    )
    print(f"[필터] 유효 TLE: {len(filtered_tle)} / {len(tle_all)}")

    # ─── Primary 위성 선택 / Select primary satellites ───
    def _find_primary(norad_id):
        """특정 NORAD ID의 TLE를 filtered_tle에서 또는 DB에서 fallback 검색"""
        sel = [t for t in filtered_tle if t[0] == norad_id]
        if not sel:
            rows = db_manager.get_latest_TLE_asof(norad_id, analysis_start)
            if not rows:
                rows = db_manager.get_latest_TLE(norad_id)
            if rows:
                sel = [(r[0], r[1], r[2], r[3]) for r in rows]
        return sel

    if args.norad:
        primary_tles = _find_primary(args.norad)
    elif args.profile_one is not None:
        primary_tles = _find_primary(args.profile_one)
    else:
        primary_tles = [t for t in filtered_tle if t[0] in primary_norad_list]
    print(f"[분석] 분석 대상 primary: {len(primary_tles)}개")

    if args.dryrun:
        print('[완료] Dry run — 데이터 준비 완료, 분석 실행 안 함')
        sys.exit(0)

    # ─── 전송/저장 설정 / Transmission setup ───
    sender = None if args.no_send else CAEventSender(API_BASE_URL, batch_size=BATCH_SIZE)
    if sender:
        print(f"[전송] API 전송 활성: url={API_BASE_URL}, batch={BATCH_SIZE}")
    else:
        print("[전송] API 전송 비활성 (--no-send)")
    if args.no_db:
        print("[저장] DB 저장 비활성 (--no-db)")

    # ─── 분석 실행 / Run analysis ───
    run_start = datetime.now(timezone.utc)
    print(f"\n{'='*60}")
    print(f"  분석 시작: {run_start.isoformat()}")
    print(f"{'='*60}")

    results_all = []
    if args.profile_one is not None:
        # 프로파일링 모드 (단일 primary, 순차 실행, 타이밍 로그 출력)
        if not primary_tles:
            print("[오류] 프로파일링 대상 primary TLE 없음")
            sys.exit(1)
        from CA.processor import process_primary_profiled
        for ref_sat in primary_tles:
            res = process_primary_profiled(
                ref_sat, filtered_tle, analysis_start, analysis_end,
                min_dt=float(args.min_dt), refine_step_s=float(args.refine_step),
            )
            results_all.extend(res)
            print(f"  → Primary {ref_sat[0]}: {len(res)} events")
    else:
        # 병렬 처리 모드 (ProcessPoolExecutor)
        with ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(
                    process_primary, ref_sat, filtered_tle,
                    analysis_start, analysis_end,
                    float(args.min_dt), float(args.refine_step),
                ): ref_sat[0]
                for ref_sat in primary_tles
            }
            total = len(futures)
            done_count = 0
            for future in as_completed(futures):
                res = future.result()
                results_all.extend(res)
                done_count += 1
                norad_id = futures[future]
                if done_count == 1 or done_count % 5 == 0 or done_count == total:
                    print(f"  진행: {done_count}/{total} | primary {norad_id}: {len(res)} events | 누적 {len(results_all)}")

    # ─── 중복 제거 / Deduplication ───
    raw_count = len(results_all)
    deduped = dedupe_events(results_all, window_s=300)
    print(f"\n[결과] 이벤트: {raw_count} (원본) → {len(deduped)} (중복 제거)")

    # ─── DB 저장 / Save to database ───
    db_saved = 0
    if not args.no_db and deduped:
        for e in deduped:
            try:
                sat1 = e.get("SAT_1", {}) if isinstance(e.get("SAT_1"), dict) else {}
                sat2 = e.get("SAT_2", {}) if isinstance(e.get("SAT_2"), dict) else {}
                db_manager.insert_CA(
                    norad1=int(sat1.get("SAT_ID", 0) or 0),
                    norad2=int(sat2.get("SAT_ID", 0) or 0),
                    name1=sat1.get("SAT_Name", ""),
                    name2=sat2.get("SAT_Name", ""),
                    tca=e.get("TCA"),
                    closest_distance_km=float(e.get("MIN_RNG", 0)),
                    probability=float(e.get("probability", 0)),
                    creation_date=e.get("Creation_date"),
                )
                db_saved += 1
            except Exception as ex:
                logging.error("DB 저장 실패: %s", ex)
        print(f"[저장] DB: {db_saved}/{len(deduped)} 건 저장 완료")

    # ─── API 전송 / Send to API server ───
    api_sent, api_fail = 0, 0
    if sender and deduped:
        if sender.batch_size > 0:
            # 배치 전송 모드
            batch = []
            for e in deduped:
                batch.append(e)
                if len(batch) >= sender.batch_size:
                    if sender.send_batch(batch):
                        api_sent += len(batch)
                    else:
                        api_fail += len(batch)
                    batch = []
            if batch:
                if sender.send_batch(batch):
                    api_sent += len(batch)
                else:
                    api_fail += len(batch)
        else:
            # 개별 전송 모드
            for e in deduped:
                if sender.send_event(e):
                    api_sent += 1
                else:
                    api_fail += 1
        print(f"[전송] API: {api_sent} 성공, {api_fail} 실패 / {len(deduped)} 총")

    # ─── 완료 / Finalize ───
    run_end = datetime.now(timezone.utc)
    duration_s = (run_end - run_start).total_seconds()
    # 분석 시간 DB 기록 (system_status 테이블)
    try:
        db_manager.update_last_analysis_time(run_end)
    except Exception:
        pass

    print(f"\n{'='*60}")
    print(f"  분석 완료: {run_end.isoformat()}")
    print(f"  소요시간: {duration_s:.1f}초")
    print(f"  최종 이벤트: {len(deduped)}건 (DB {db_saved}건, API {api_sent}건)")
    print(f"{'='*60}")
