# propagation.py

import numpy as np
from datetime import datetime
from astropy.time import Time
from scipy.integrate import solve_ivp

def propagate_with_scipy(start_time_utc, end_time_utc, output_step_sec, 
                         y0, force_model, rtol=1e-12, atol=1e-12):
    """
    [SciPy DOP853 사용]
    시작/종료 시각(datetime)을 입력받아 궤도를 전파하고 결과를 저장합니다.
    """
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
        t_span=t_span,
        y0=y0,
        method='DOP853',
        t_eval=t_eval,
        rtol=rtol,
        atol=atol
    )

    print("궤도 전파 완료.")
    
    # 4. 결과 정리
    ephemeris_table = np.column_stack((solution.t, solution.y.T))
    return ephemeris_table