"""
ConjunctionAssessment.py

위성 근접 위험(Conjunction) 자동 평가 메인 스크립트

[기능 요약]
- TLE 데이터베이스에서 전체 TLE 불러오기 및 유효기간 필터링
- 고도/궤도 경로/시간적 근접성 등 단계별 필터링
- SGP4 기반 실제 최소 접근 거리 평가
- 결과 구조화 및 DB 저장 준비

[입력]
- TLE 데이터: TLEmanager에서 불러옴 (all_tles)
- 분석 시작/종료 시각: now_epoch, analysis_end
- 필터 기준: ANALYSIS_DURATION, TLE_AGE_LIMIT 등

[처리 순서]
1. 전체 TLE 데이터 로드 및 개수 확인
2. 유효기간 지난 TLE 제거
3. 고도 기준 후보 추림
4. 궤도 경로 유사성 기준 후보 추림
5. 시간적 근접성 기준 후보 추림
6. 실제 최소 접근 거리 기반 근접 위험 평가
7. 결과 구조화 및 DB 저장 준비

[출력]
- 각 근접 위험 이벤트별:
    - 위성 정보, 근접 시각, 최소 거리, 충돌 확률, 상대 위치/속도 등

[사용법]
- main() 함수 없이 스크립트 실행 시 전체 분석 수행
- 각 단계별 print로 진행 상황 및 결과 확인 가능
"""

from MISC.DBmanager import DBmanager
from MISC.TLEmanager import TLEmanager
from CA.CA_filter import CA_filter
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
import MISC.Structurer as structer

# ==========================
# 주요 설정값
# ==========================
ANALYSIS_DURATION = 1       # [days] 분석 기간
TLE_AGE_LIMIT = 10          # [days] TLE 유효기간
# ==========================


# Step 1: TLE 데이터 관리 객체 및 근접 필터링 객체 생성
tle_manager = TLEmanager()
CA_filter = CA_filter()
db_manager = DBmanager()


# Step 2: 전체 TLE 개수 및 데이터 조회
count_tle = db_manager.get_tle_count()  # DB 내 TLE 총 개수
tle_all = tle_manager.all_tles()         # 모든 TLE 데이터 로드
print(f"Total TLE records in database: {count_tle}") 

# Step 2-1: primary 위성 리스트 DB에서 불러오기 (DBmanager 메서드 사용)
primary_norad_list = db_manager.get_primary_norad_list()

# Step 3: 분석 기준 시각 설정
now_epoch = datetime.now()             # 현재 시각을 기준 epoch로 사용
analysis_start = now_epoch            # 분석 시작 시각
analysis_end = now_epoch + timedelta(days=ANALYSIS_DURATION)    # 분석 종료 시각 (1일 후)

# Step 4: 1차 필터 - 유효기간 지난 TLE 제거
filtered_tle = tle_manager.filter_outdated_tles(
    tle_all, analysis_start, analysis_end, pad_days=TLE_AGE_LIMIT
)
print(f"1st Filtered TLE records: {len(filtered_tle)}")  # 필터링 후 TLE 개수 출력

# Step 4-1: primary 위성만 추출
primary_tles = [tle for tle in filtered_tle if tle[0] in primary_norad_list]
print(f"Primary TLE records: {len(primary_tles)}")


# ==========================
# 함수 정의 (상단에 위치)
# ==========================
def process_primary(i):
    db_manager = DBmanager()  # 각 프로세스마다 새로 생성
    ref_line2 = primary_tles[i][2]
    remain_tle = CA_filter.filter_altitude(filtered_tle, ref_line2, pad=0)
    remain_tle = CA_filter.filter_orbitpath(remain_tle, ref_line2)
    ref_sat = primary_tles[i]
    remain_events = CA_filter.filter_time(
        ref_sat, remain_tle, analysis_days=10, time_window=300.0, d_tol_km=100.0
    )
    ca_res = CA_filter.fine_filter_min_distance(
        ref_sat, remain_tle, remain_events, dt_s=1.0
    )
    results = []
    for r in ca_res:
        orbit1 = structer.reassemble_orbit(r['sat1_ephem'])
        orbit2 = structer.reassemble_orbit(r['sat2_ephem'])
        sat1_info = db_manager.get_SATCAT_info(r['sat1_norad'])
        sat2_info = db_manager.get_SATCAT_info(r['sat2_norad'])
        SAT_1 = structer.SAT_struc(
            r['sat1_norad'], sat1_info["OBJECT_NAME"], sat1_info["OBJECT_TYPE"], 0, orbit1, 0, sat1_info["RCS"]
        )
        SAT_2 = structer.SAT_struc(
            r['sat2_norad'], sat2_info['OBJECT_NAME'], sat2_info['OBJECT_TYPE'], 0, orbit2, 0, sat2_info['RCS']
        )
        COLLI = {
            "CDM_ID"        : f"{r['sat1_norad']}_{r['sat2_norad']}_{r['closest_time']}",
            "Creation_date" : datetime.now(),
            "TCA"           : r['closest_time'],
            "MIN_RNG"       : r['closest_distance_km'],
            "probability"   : r['probability'],
            "SAT_1"         : SAT_1,
            "SAT_2"         : SAT_2,
            "COLLI_Info"    : structer.COLLI_Info(
                r['rel_vec'][0:3], r['rel_vec'][3:6]
            )
        }
        db_manager.insert_CA(
            r['sat1_norad'], r['sat2_norad'], sat1_info['OBJECT_NAME'], sat2_info['OBJECT_NAME'], r['closest_time'], r['closest_distance_km'], r['probability']
        )
        results.append(COLLI)
    return results


# ==========================
# 메인 실행 로직 (하단에 위치)
# ==========================
if __name__ == "__main__":
    # 병렬처리 적용: 각 primary object별로 process_primary 함수 실행
    results_all = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_primary, i) for i in range(len(primary_tles))]
        for future in as_completed(futures):
            res = future.result()
            results_all.extend(res)

    # 분석이 모두 끝난 후 마지막 분석 시간 기록 (system_status 테이블에 저장)
    db_manager.update_last_analysis_time(datetime.now())
