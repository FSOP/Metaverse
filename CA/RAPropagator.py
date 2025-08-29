# propagation.py

import numpy as np
from datetime import datetime
from astropy.time import Time
from scipy.integrate import solve_ivp

def altitude_event(t, y):
    r = y[:3]
    alt = np.linalg.norm(r) / 1000.0 - 6378.137  # km
    # print(alt)
    return alt  # 0이 되는 순간 이벤트 발생

altitude_event.terminal = True
# altitude_event.direction = -1  # 하강 방향에서만 이벤트 발생

def propagate_with_scipy(epoch, analysis_period, output_step_sec, 
                         y0, force_model, rtol=1e-12, atol=1e-12):
    """
    [SciPy DOP853 사용]
    시작/종료 시각(datetime)을 입력받아 궤도를 전파하고 결과를 저장합니다.
    """
    start_time_utc = analysis_period[0]
    end_time_utc = analysis_period[1]

    # 1. ForceModel에 시뮬레이션 시작 시점(MJD)을 설정합니다.
    start_mjd = Time(start_time_utc).mjd
    force_model.aux_params['Mjd_UTC'] = start_mjd

    # 2. 적분 구간과 결과를 저장할 시간 배열을 계산합니다.
    duration_sec = (end_time_utc - start_time_utc).total_seconds()
    t_span = [0.0, duration_sec]
    t_eval = np.arange(0.0, duration_sec + 1, output_step_sec)
    
    print("SciPy DOP853 적분기로 궤도 전파를 시작합니다...")

    # 3. solve_ivp 함수 호출
    solution = solve_ivp(
        fun=force_model,
        t_span=t_span,  # 적분 구간
        y0=y0,          # 초기 상태 벡터
        method='DOP853',# 적분 방법
        t_eval=t_eval,  # 출력 시간 배열
        rtol=rtol,      # 상대 오차 허용치
        atol=atol,      # 절대 오차 허용치
        events=altitude_event   # 고도 이벤트
    )

    print("궤도 전파 완료.")
    
    # 4. 결과 정리
    ephemeris_table = np.column_stack((solution.t, solution.y.T))
    return ephemeris_table