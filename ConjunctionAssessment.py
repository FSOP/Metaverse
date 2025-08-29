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

# Step 2: 전체 TLE 개수 및 데이터 조회
count_tle = DBmanager().get_tle_count()  # DB 내 TLE 총 개수
tle_all = tle_manager.all_tles()         # 모든 TLE 데이터 로드
print(f"Total TLE records in database: {count_tle}") 

# Step 3: 분석 기준 시각 설정
now_epoch = datetime.now()             # 현재 시각을 기준 epoch로 사용
analysis_start = now_epoch            # 분석 시작 시각
analysis_end = now_epoch + timedelta(days=ANALYSIS_DURATION)    # 분석 종료 시각 (1일 후)

# Step 4: 1차 필터 - 유효기간 지난 TLE 제거
filtered_tle = tle_manager.filter_outdated_tles(
    tle_all, analysis_start, analysis_end, pad_days=TLE_AGE_LIMIT
)
print(f"1st Filtered TLE records: {len(filtered_tle)}")  # 필터링 후 TLE 개수 출력

# Step 5: 각 TLE(위성)별로 근접 위험 분석 수행
for i in range(len(filtered_tle)):
    ref_line2 = filtered_tle[i][2]  # 기준 위성의 TLE line2 (고도 비교용)

    # Step 5-1: 2차 필터 - 고도 기준 근접 후보 추림
    remain_tle = CA_filter.filter_altitude(filtered_tle[i+1:], ref_line2, pad=0)
    # print(f"2nd Filtered TLE records: {len(remain_tle)}")  # 고도 필터링 후 개수

    # Step 5-2: 3차 필터 - 궤도 경로 유사성 기준 후보 추림
    remain_tle = CA_filter.filter_orbitpath(remain_tle, ref_line2)
    # print(f"3rd Filtered TLE records: {len(remain_tle)}")  # 궤도 경로 필터링 후 개수

    # Step 5-3: 4차 필터 - 시간적 근접성 기준 후보 추림
    ref_sat = filtered_tle[i]  # 기준 위성 정보
    remain_events = CA_filter.filter_time(
        ref_sat, remain_tle, analysis_days=10, time_window=300.0, d_tol_km=100.0
    )
    # print(f"4th Filtered TLE records: {len(remain_tle)}")  # 시간 필터링 후 개수

    # Step 5-4: 5차 필터 - 실제 최소 접근 거리 기반 근접 위험 평가
    ca_res = CA_filter.fine_filter_min_distance(
        ref_sat, remain_tle, remain_events, dt_s=1.0
    )
    # print(f"5th Filtered TLE records: {len(ca_res)}")  # 근접 위험 평가 후 개수

    # Step 6: 근접 위험 결과를 구조화 및 DB 저장 준비
    for r in ca_res:
        # 위성 궤도 정보 복원
        orbit1 = structer.reassemble_orbit(r['sat1_ephem'])
        orbit2 = structer.reassemble_orbit(r['sat2_ephem'])
        # 위성 구조체 생성
        SAT_1 = structer.SAT_struc(
            r['sat1_norad'], "SAT1_NAME", "SAT_TYPE", 0, orbit1, 0, "MASS"
        )
        SAT_2 = structer.SAT_struc(
            r['sat2_norad'], "SAT2_NAME", "SAT_TYPE", 0, orbit2, 0, "MASS"
        )

        # 근접 위험(COLLI) 정보 구조화
        COLLI = {
            "CDM_ID"        : f"{r['sat1_norad']}_{r['sat2_norad']}_{r['closest_time']}",  # 고유 ID
            "Creation_date" : datetime.now(),
            "TCA"           : r['closest_time'],           # 근접 시각
            "MIN_RNG"       : r['closest_distance_km'],    # 최소 접근 거리
            "probability"   : r['probability'],            # 충돌 확률
            "SAT_1"         : SAT_1,                       # 위성1 정보
            "SAT_2"         : SAT_2,                       # 위성2 정보
            "COLLI_Info"    : structer.COLLI_Info(
                r['rel_vec'][0:3], r['rel_vec'][3:6]
            )  # 상대 위치/속도 정보
        }

        # 실제 DB 저장 예시 (주석 처리)
        # DBmanager().insert_CA(
        #     r['sat1_norad'], r['sat2_norad'], "NONAME1", "NONAME2", r['closest_time'], r['closest_distance_km']
        # )
