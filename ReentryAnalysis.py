"""
ReentryAnalysis.py

위성 재진입(추락) 분석 메인 스크립트

[기능 요약]
- TLE 데이터를 불러와서, 유효 기간/궤도/드래그(Bstar) 기준으로 필터링
- SGP4로 100~170km 고도 도달 시점 및 상태벡터 계산
- HPOP(고정밀 궤도 전파)로 지표면 추락 시점/위치 계산
- 결과를 dictionary로 저장 및 요약 출력

[입력]
- TLE 데이터: TLEmanager에서 불러옴 (all_tles)
- 분석 시작/종료 시각: start_epoch, end_epoch
- 필터 기준: ALTITUDE_THRESHOLD(170km), BSTAR_THRESHOLD(0.05)
- HPOP 관련 상수/파라미터: CONSTANTS, AUX_PARAMS, ephemeris/gravity 파일 등

[처리 순서]
1. TLE 데이터 전체 불러오기
2. 유효 기간 내 TLE만 필터링
3. 궤도(perigee) 기준 필터링
4. Bstar(드래그) 기준 필터링
5. SGP4로 각 위성의 200km 도달 시점/상태벡터 계산
6. HPOP으로 각 위성의 지표면 추락 시점/위치 계산
7. 결과를 dictionary로 저장 및 요약 출력

[출력]
- 각 위성별:
    - NORAD 번호
    - SGP4 분석 코드(0: 정상, 1: 오류, 2: 이미 추락)
    - 200km 도달 시각/상태벡터
    - HPOP 기반 지표면 추락 시각/위치
- 전체 요약: result_impact 리스트

[사용법]
- 필요한 TLE 데이터와 HPOP 관련 파일이 프로젝트 루트에 있어야 함
- main() 함수 실행 시 자동으로 전체 분석 수행 및 결과 출력

[참고]
- SGP4/CA_filter/HPOP 관련 모듈은 별도 구현 필요
- 각 단계별 print로 진행 상황 및 결과 확인 가능
"""

import os, sys
import MISC.Structurer as structer
import numpy as np

from datetime import datetime, timedelta
from HPOP.force_models import ForceModel
from MISC.TLEmanager import TLEmanager
from MISC.DBmanager import DBmanager
from CA.CA_filter import CA_filter
from CA.RA_ephemeris import propagate_hpop_to_surface
from CA.orbitcalculator import orcal
from HPOP.force_models import ForceModel
from HPOP.eop import EOPManager
from HPOP.astroephemeris import EphemerisManager
from HPOP.gravity_models import GravityModel # 새로 만든 클래스를 import
from HPOP.atmosphere import AtmosphereModel


ALTITUDE_THRESHOLD = 200.0  # km
BSTAR_THRESHOLD = 0.05

def main():
    tle_manager = TLEmanager() # TLEmanager: TLE 데이터 관리 및 필터링
    CA_manager = CA_filter() # CA_filter: 궤도 필터링 및 SGP4 기반 분석
    db_man = DBmanager() # DBmanager: 위성 정보 DB 연동
    or_cal = orcal() # orcal: 궤도 계산기

    total_res = []

    start_epoch = datetime.now()
        # 분석 시작/종료 시각 설정 (현재~10일 후)
    end_epoch = start_epoch + timedelta(days=10)

    # Step 1: Load TLEs
        # 전체 TLE 데이터 불러오기
    all_tles = tle_manager.all_tles()

    # Step 2: Filter outdated TLEs
        # 유효 기간 내 TLE만 필터링 (pad_days: 여유 기간)
    tles = tle_manager.filter_outdated_tles(all_tles, start_epoch, end_epoch, pad_days=10)

    # Step 3: Filter by apogee/perigee
    # 궤도(perigee) 기준 필터링 (저고도 위성만 선별)
    tles = CA_manager.filter_perigee(tles, ALTITUDE_THRESHOLD)

    # Step 4: Filter by Bstar
        # Bstar(드래그) 기준 필터링 (대기저항 큰 위성만 선별)
    tles = CA_manager.filter_BSTAR(tles, BSTAR_THRESHOLD)  # example value

    print(f"Number of whole TLE: {len(all_tles)} Filtered TLEs: {len(tles)}")
    pass

    # Step 5: For each filtered TLE, analyze reentry
    filtered_result = []
    for tle in tles:
        sat = tle
            # SGP4로 ALTITUDE_THRESHOLD(저고도) 도달 시각/상태벡터 계산
        code, reentry_time, state_vector = CA_manager.get_state_at_altitude(sat, start_epoch, ALTITUDE_THRESHOLD, 30, 10)
        reenter_sat = {
            "norad": sat[0],
            "code": code,  # 0: success, 1: error, 2: already-decayed
            "reentry_time": reentry_time,
            "state_vector": state_vector
        }
        filtered_result.append(reenter_sat)
        print(f"Satellite {sat[0]}: code={code}, reentry_time={reentry_time}, state={state_vector}")

    # AUX_PARAMS: HPOP에서 사용할 외력/환경 모델 파라미터
    AUX_PARAMS = {
    'mass': 1000.0, 'area_drag': 10.0, 'area_solar': 10.0,
    'Cd': 2.35, 'Cr': 1.0, 'n_max': 70, 'm_max': 70,
    'sun': False, 'moon': False, 'sRad': False, 'drag': True        
    }

    # 프로젝트 루트 경로 및 외부 파일 경로 지정
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    ephem_file = os.path.join(PROJECT_ROOT, 'MISC', 'de440.bsp')        
    gravity_file = os.path.join(PROJECT_ROOT, 'MISC', 'EGM2008.gfc')

    force_model = ForceModel(
        aux_params=AUX_PARAMS,
        eop_manager=EOPManager(),
        ephem_manager=EphemerisManager(ephem_file),
        gravity_model=GravityModel(gravity_file, n_max=AUX_PARAMS['n_max'], m_max=AUX_PARAMS['m_max']),
        atmosphere_model=AtmosphereModel()
        # ForceModel: HPOP에서 사용할 외력/환경 모델 객체 생성
    )

    # 1차 필터링한 결과 나온 event들을 정밀 궤도전파기로 정밀분석
    for reenter_sat in filtered_result:
        if reenter_sat['code'] == 0 and reenter_sat['state_vector'] is not None:
            # HPOP을 이용해 위성이 지표면에 도달할 때까지 propagate
            impact_time, impact_location, ephemeris = propagate_hpop_to_surface(reenter_sat['state_vector'], reenter_sat['reentry_time'], force_model)

            # 위성에 대한 정보
            sat_info = db_man.get_SATCAT_info(reenter_sat['norad'])
            ephem1 = structer.reassemble_orbit(ephemeris)
            SAT1 = structer.SAT_struc(sat_info['NORAD_CAT_ID'], sat_info['OBJECT_NAME'], sat_info['OBJECT_TYPE'], sat_info['RCS'], ephem1, 0, "")

            inst_eop = force_model.get_eop()
            inst_eop['Date'] = start_epoch.strftime('%Y%m%d')
            inbound_info = structer.inbound_info(or_cal.cal_inbound(ephemeris), inst_eop)

            total_res.append(structer.crash(
                crash_id = f"{impact_time.strftime('%Y%m%d_%H%M%S')}_{reenter_sat['norad']}",
                creation_date = datetime.now(),
                start_time = reenter_sat['reentry_time'],
                SAT_info = SAT1,
                inbound_info = inbound_info,
                orbit_crash = ephem1,
                time_crash = impact_time,
                point_crash = impact_location,
                prob_crash = None
            ))


if __name__ == "__main__":
    main()