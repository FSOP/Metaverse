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

from datetime import datetime, timedelta, timezone
from MISC.TLEmanager import TLEmanager
from MISC.DBmanager import DBmanager
from CA.CA_filter import CA_filter
from CA.RA_ephemeris import propagate_hpop_to_surface
from CA.orbitcalculator import orcal
from HPOP.eop import EOPManager
from HPOP.astroephemeris import EphemerisManager
from HPOP.gravity_models import GravityModel # 새로 만든 클래스를 import
from HPOP.atmosphere import AtmosphereModel
from HPOP.force_models import ForceModel
from CA.reentry_processor import process_reentry_sat
from MISC import reentry_uploader as ephem_client
import json


ALTITUDE_THRESHOLD = 200.0  # km
BSTAR_THRESHOLD = 0.05

def main():
    tle_manager = TLEmanager() # TLEmanager: TLE 데이터 관리 및 필터링
    CA_manager = CA_filter() # CA_filter: 궤도 필터링 및 SGP4 기반 분석
    db_man = DBmanager() # DBmanager: 위성 정보 DB 연동
    or_cal = orcal() # orcal: 궤도 계산기

    # results are uploaded to server; no local accumulation

    start_epoch = datetime.now(timezone.utc)
        # 분석 시작/종료 시각 설정 (현재~10일 후)
    end_epoch = start_epoch + timedelta(days=10)

    # Step 1: Load TLEs 전체 TLE 데이터 불러오기
    all_tles = tle_manager.all_tles()
 
    # Step 2: Filter outdated TLEs  # 유효 기간 내 TLE만 필터링 (pad_days: 여유 기간)
    tles = tle_manager.filter_outdated_tles(all_tles, start_epoch, end_epoch, pad_days=10)

    # Step 3: Filter by apogee/perigee  # 궤도(perigee) 기준 필터링 (저고도 위성만 선별)
    tles = CA_manager.filter_perigee(tles, ALTITUDE_THRESHOLD)

    # Step 4: Filter by Bstar  # Bstar(드래그) 기준 필터링 (대기저항 큰 위성만 선별)
    tles = CA_manager.filter_BSTAR(tles, BSTAR_THRESHOLD)  # example value

    print(f"Number of whole TLE: {len(all_tles)} Filtered TLEs: {len(tles)}")
    

    # Step 5: For each filtered TLE, analyze reentry
    filtered_result = []
    # no local accumulation of results
    for tle in tles:
        sat = tle
            # SGP4로 ALTITUDE_THRESHOLD(저고도) 도달 시각/상태벡터 계산
        code, reentry_time, state_vector = CA_manager.get_state_at_altitude(sat, start_epoch, ALTITUDE_THRESHOLD, 30, 10)
        # Only keep entries with successful state vector (code == 0)
        if code != 0 or state_vector is None:
            print(f"Satellite {sat[0]} skipped (code={code}, state_vector={state_vector})")
            continue
        reenter_sat = {
            "norad": sat[0],
            "code": code,  # 0: success, 1: error, 2: already-decayed
            "reentry_time": reentry_time,
            "state_vector": state_vector,
            "tle_line1": sat[1] if len(sat) > 1 else None,
            "tle_line2": sat[2] if len(sat) > 2 else None,
            "tle_epoch": sat[3] if len(sat) > 3 else None,
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
    for sat in filtered_result:
        # pass the full dict so `process_reentry_sat` can use the precomputed 'state_vector' path
        processed = process_reentry_sat(sat, start_epoch, CA_manager, force_model, db_man, or_cal)
        if not processed:
            continue

        # Serialize & upload sampled ephemeris to server; fallback to repr()
        try:
            sampled = processed.get('ephemeris_sampled') or []
            crash_id_local = processed.get('crash_id')
            # build metadata to send along with ephemeris file so server can store searchable fields
            metadata = {
                'crash_id': processed.get('crash_id'),
                'norad_cat_id': processed.get('norad_cat_id'),
                'object_name': processed.get('object_name'),
                'object_type': processed.get('object_type'),
                'rcs': processed.get('rcs'),
                'time_crash': processed.get('time_crash').astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ') if processed.get('time_crash') is not None else None,
                'creation_date': processed.get('creation_date').astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ') if processed.get('creation_date') is not None else None,
                'prob_crash': processed.get('prob_crash'),
                'coordinate_frame': 'ICRF',
                'frame_type': 'inertial',
            }

            # 초기 상태(epoch)를 명시적으로 포함: processed의 start_time 우선, 없으면 sat의 reentry_time 사용
            initial_epoch = None
            if processed.get('start_time') is not None:
                initial_epoch = processed.get('start_time')
            else:
                initial_epoch = sat.get('reentry_time')

            debug_metadata = {
                "norad": sat.get("norad"),
                "tle_line1": sat.get("tle_line1"),
                "tle_line2": sat.get("tle_line2"),
                "tle_epoch": sat.get("tle_epoch"),
                "initial_state_vector": sat.get("state_vector"),
                "initial_state_epoch": initial_epoch.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ') if initial_epoch is not None else None,
                "reentry_time": sat.get("reentry_time"),
                "hpop_step_seconds": 60,
                "hpop_max_duration_hours": 12,
                "coordinate_frame": "ICRF",
                "frame_type": "inertial",
            }

            orbit_ref = ephem_client.upload_sampled_ephemeris(
                sampled,
                prefer='msgpack',
                compress=True,
                timeout=30,
                max_retries=3,
                crash_id=crash_id_local,
                use_multipart=True,
                metadata=metadata,
                debug_metadata=debug_metadata,
            )
            upload_status = 'success'
        except Exception as e:
            orbit_ref = repr(processed.get('ephemeris_sampled'))
            upload_status = f'failed: {e!r}'

        inbound_info = processed.get('inbound_info_str') or ''
        try:
            orbit_summary = processed.get('orbit_summary') or ephem_client.build_orbit_summary(processed.get('ephemeris_sampled') or [])
            combined = {'inbound': inbound_info, 'orbit_summary': orbit_summary}
            inbound_info_str = json.dumps(combined, ensure_ascii=False)
        except Exception:
            inbound_info_str = inbound_info

        # 로그: 업로드 결과 및 충돌 검출 여부 표시
        time_crash = processed.get('time_crash')
        crash_id = processed.get('crash_id')
        noimpact = time_crash is None
        print(f"Upload result: crash_id={crash_id} noimpact={noimpact} upload_status={upload_status} orbit_ref={orbit_ref}")



if __name__ == "__main__":
    main()