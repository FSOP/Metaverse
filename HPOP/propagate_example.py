# simulation.py

import numpy as np
from force_models import ForceModel
from eop import EOPManager
from astroephemeris import EphemerisManager
from gravity_models import GravityModel # 새로 만든 클래스를 import
from atmosphere import AtmosphereModel

import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from propagator import propagate_with_scipy

CONSTANTS = {
    'GM_Earth': 3.986004418e14,     # m^3/s^2
    'GM_Sun': 1.32712440018e20,
    'GM_Moon': 4.9048695e12,
    'R_Earth': 6378137.0,            # 지구 평균 반경 [m]
    'AU': 149597870700.0,            # 천문단위 [m]
    'P_Sol': 4.56e-6,                # 1 AU에서의 태양 압력 [N/m^2]
    'omega_Earth': 7.292115e-5       # 지구 자전 각속도 [rad/s]
}

# # 시뮬레이션 파라미터 및 위성 제원
# AUX_PARAMS = {
#     'Mjd_UTC': 58849.0,              # 시뮬레이션 시작 시점 (2020-01-01 00:00 UTC)
#     'mass': 1000.0,                  # 위성 질량 [kg]
#     'area_drag': 10.0,               # 대기 항력 계산용 면적 [m^2]
#     'area_solar': 10.0,              # 태양복사압 계산용 면적 [m^2]
#     'Cd': 2.2,                       # 항력 계수
#     'Cr': 1.8,                       # 태양복사압 반사 계수
#     'n_max': 70,                     # 지구 중력장 모델 최대 차수
#     'm_max': 70,
#     # 섭동 모델 활성화 플래그
#     'sun': False,
#     'moon': False,
#     'sRad': False,
#     'drag': False,
#     'planets': False,
#     'Relativity': False
# }

# 시뮬레이션 파라미터 (Mjd_UTC는 이제 propagation 함수가 설정하므로 제거)
AUX_PARAMS = {
    'mass': 1000.0, 'area_drag': 10.0, 'area_solar': 10.0,
    'Cd': 2.35, 'Cr': 1.0, 'n_max': 70, 'm_max': 70,
    'sun': False, 'moon': False, 'sRad': False, 'drag': True
}

if __name__ == "__main__":
    # easyHPOP_handle.py에 더 간편한 HPOP사용 예시가 있음

    # (a) 데이터 파일 경로 (이전과 동일)
    ephem_file = './metaverse/de440.bsp'
    gravity_file = './metaverse/EGM2008.gfc'    
    
    # (b) 위성 초기 상태 벡터 (이전과 동일)
    r0 = np.array([-1939810, 5568630, 3540650])
    v0 = np.array([-5459.92, -4067.29, 3405.61])
    y_initial = np.hstack((r0, v0))

    # (c) 궤도 전파 기간을 datetime으로 설정
    start_time = datetime(2025, 8, 17, 11, 0, 0)  # 2025년 8월 17일 11:00
    duration_min = 60*12
    end_time = start_time + timedelta(seconds= duration_min*60)
    output_step_seconds = 1 # 60초 간격으로 결과 저장

    # (d) 모든 관리자 객체 생성 (이전과 동일)
    eop_manager = EOPManager()
    ephem_manager = EphemerisManager(ephem_file)
    gravity_model = GravityModel(gravity_file, n_max=AUX_PARAMS['n_max'], m_max=AUX_PARAMS['m_max'])
    atmosphere_model = AtmosphereModel()

    # (e) ForceModel 객체 생성. Mjd_UTC는 여기서 전달하지 않습니다.
    force_model = ForceModel(
        consts=CONSTANTS, 
        aux_params=AUX_PARAMS, 
        eop_manager=eop_manager,
        ephem_manager=ephem_manager,
        gravity_model=gravity_model,
        atmosphere_model=atmosphere_model
    )
    
    # (f) datetime을 사용하는 SciPy 전파 함수 호출
    ephemeris = propagate_with_scipy(
        start_time, end_time, output_step_seconds,
        y_initial, force_model, rtol=1e-12, atol=1e-12
    )

    # (g) 결과 출력 및 시각화 (이전과 동일)
    print("\n--- Generated Ephemeris (first 10 rows) ---")
    print("Time(s)      X(km)         Y(km)         Z(km)         Vx(km/s)      Vy(km/s)      Vz(km/s)")
    for row in ephemeris:
        print(f"{row[0]:<12.2f} {row[1]/1000:<12.3f} {row[2]/1000:<12.3f} {row[3]/1000:<12.3f} "
              f"{row[4]/1000:<12.4f} {row[5]/1000:<12.4f} {row[6]/1000:<12.4f}")
    
    # Plotting
    x_coords_km = ephemeris[:, 1] / 1000.0
    y_coords_km = ephemeris[:, 2] / 1000.0

    # plt.figure(figsize=(10, 10))
    # plt.plot(x_coords_km, y_coords_km, label=f'Orbit from {start_time.strftime("%Y-%m-%d %H:%M")}')
    # plt.plot(0, 0, 'bo', markersize=12, label='Earth')
    # plt.xlabel('X (km) [ICRS]'), plt.ylabel('Y (km) [ICRS]'), plt.title('High-Precision Satellite Orbit')
    # plt.grid(True), plt.axis('equal'), plt.legend(), plt.show()