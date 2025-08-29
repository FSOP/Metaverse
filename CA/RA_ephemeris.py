import numpy as np
from datetime import timedelta
# from CA.RAPropagator import propagate_with_scipy
from HPOP.propagator import propagate_with_scipy
from HPOP.easyHPOP_handle import HPOP_handle


def propagate_hpop_to_surface(state_vector, start_time, force_model, max_duration_hours=12, step_seconds=60):
    """
    HPOP을 이용해 위성이 지표면(alt <= 0km)에 도달할 때까지 궤도 예측을 수행합니다.
    Args:
        state_vector: 초기 상태벡터 (x, y, z, vx, vy, vz), 단위 m 또는 km
        start_time: 시작 시각 (datetime)
        force_model: HPOP에서 사용할 힘 모델 객체
        max_duration_hours: 최대 propagate 시간 (시간)
        step_seconds: propagate 간격 (초)
    Returns:
        impact_time: 지표면 도달 시각 (datetime)
        impact_location: 지표면 도달 위치 (km, 3차원 벡터)
        alt > 0이면 None 반환
    """
    y = np.array(state_vector)  # 상태벡터 복사
    # 입력이 km 단위면 m로 변환 (궤도 예측은 m 단위 사용)
    if np.abs(y[:3]).max() < 10000:
        y[:3] *= 1000
        y[3:] *= 1000
    
    # HPOP propagate 수행 (scipy ODE 사용)
    ephemeris = propagate_with_scipy(start_time, (start_time, start_time + timedelta(hours=max_duration_hours)), 60, y, force_model, rtol=1e-3, atol=1e-3)
    
    y_impact = ephemeris[-1, 1:7]  # 마지막 시점 상태벡터
    r = y_impact[1:4]  # 위치벡터
    alt = np.linalg.norm(r)/1000 - 6378.137  # 고도(km), WGS84 지구 반경 기준
    if alt <= 0:  # 지표면 도달(또는 100km 이하)
        impact_time = start_time + timedelta(seconds=step_seconds)  # 도달 시각 추정
        impact_location = r / 1000.0  # 위치(km)
        return impact_time, impact_location, ephemeris
    
    # 아직 지표면 도달 못함
    else:
        print(f"지표면에 도달하지 않았습니다. 마지막 고도: {alt} km ")
        y = y_impact
        return None, None