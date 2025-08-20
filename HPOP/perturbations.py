# perturbations.py

import numpy as np

def accel_point_mass(r_sat_earth, r_earth_ssb, r_body_ssb, gm_body):
    """
    점질량 천체(태양, 달, 행성)가 위성에 가하는 제3체 섭동 가속도를 계산합니다.
    모든 벡터는 ICRS (관성) 좌표계 기준입니다.

    Args:
        r_sat_earth (np.ndarray): 지구 중심에서 위성까지의 위치 벡터 [m].
        r_earth_ssb (np.ndarray): 태양계 질량중심(SSB)에서 지구까지의 위치 벡터 [m].
        r_body_ssb (np.ndarray): SSB에서 섭동 천체(태양, 달 등)까지의 위치 벡터 [m].
        gm_body (float): 섭동 천체의 중력 상수 (GM) [m^3/s^2].

    Returns:
        np.ndarray: 위성에 가해지는 3차원 섭동 가속도 벡터 [m/s^2].
    """
    # 1. 섭동 천체에서 지구까지의 벡터와 위성까지의 벡터를 계산합니다.
    r_earth_body = r_earth_ssb - r_body_ssb
    r_sat_body = (r_sat_earth + r_earth_ssb) - r_body_ssb

    # 2. 각 벡터의 크기(거리)를 계산합니다.
    d_earth_body = np.linalg.norm(r_earth_body)
    d_sat_body = np.linalg.norm(r_sat_body)

    # 3. 섭동 가속도 공식을 적용합니다.
    # a_pert = GM * ( r_sat_body / |r_sat_body|^3 - r_earth_body / |r_earth_body|^3 )
    term1 = r_sat_body / (d_sat_body**3)
    term2 = r_earth_body / (d_earth_body**3)
    
    acceleration = gm_body * (term1 - term2)
    
    return acceleration

def accel_solar_radiation_pressure(r_sat_earth, r_earth_ssb, r_sun_ssb, consts, aux_params):
    """
    태양복사압(SRP)에 의한 섭동 가속도를 계산합니다.
    간단한 원통형 지구 그림자 모델을 포함합니다.

    Args:
        r_sat_earth (np.ndarray): 지구 중심에서 위성까지의 위치 벡터 [m].
        r_earth_ssb (np.ndarray): SSB에서 지구까지의 위치 벡터 [m].
        r_sun_ssb (np.ndarray): SSB에서 태양까지의 위치 벡터 [m].
        consts (dict): P_Sol, AU, R_Earth 등 물리 상수.
        aux_params (dict): Cr, area_solar, mass 등 위성 제원.

    Returns:
        np.ndarray: SRP에 의한 3차원 섭동 가속도 벡터 [m/s^2].
    """
    # 1. 태양-위성, 지구-태양 벡터 계산
    r_sun_earth = r_earth_ssb - r_sun_ssb
    r_sun_sat = (r_sat_earth + r_earth_ssb) - r_sun_ssb
    
    # 2. 그림자 판별 (원통형 모델)
    nu = 1.0  # 조명 계수 (1: 햇빛 받음, 0: 그림자)
    
    # 위성이 태양 방향을 기준으로 지구 뒤편에 있는지 확인
    if np.dot(r_sat_earth, r_sun_earth) > 0:
        # 위성이 지구-태양 라인에 얼마나 떨어져 있는지(거리) 계산
        dist_from_sun_earth_line = np.linalg.norm(np.cross(r_sat_earth, r_sun_earth)) / np.linalg.norm(r_sun_earth)
        
        # 거리가 지구 반경보다 작으면 그림자 안에 있는 것
        if dist_from_sun_earth_line < consts['R_Earth']:
            nu = 0.0
            
    # 3. SRP 가속도 계산
    if nu > 0:
        # 태양과의 거리
        d_sun_sat = np.linalg.norm(r_sun_sat)
        # 1 AU에서의 태양 압력을 현재 거리 기준으로 보정
        p_sol = consts['P_Sol'] * (consts['AU'] / d_sun_sat)**2
        # 태양에서 위성을 향하는 단위 벡터
        u_sun_sat = r_sun_sat / d_sun_sat
        
        # SRP 가속도 공식: a = -nu * P * (Cr * A/m) * u
        srp_accel = -nu * p_sol * (aux_params['Cr'] * aux_params['area_solar'] / aux_params['mass']) * u_sun_sat
    else:
        srp_accel = np.zeros(3)
        
    return srp_accel

def accel_drag(density, r_sat_itrs, v_sat_icrs, E_icrs_to_itrs, consts, aux_params):
    """
    대기 항력에 의한 섭동 가속도를 계산합니다.

    Args:
        density (float): 위성 위치에서의 대기 밀도 [kg/m^3].
        r_sat_itrs (np.ndarray): 위성 위치 (ITRS) [m].
        v_sat_icrs (np.ndarray): 위성 속도 (ICRS) [m/s].
        E_icrs_to_itrs (np.ndarray): ICRS -> ITRS 변환 행렬.
        consts (dict): omega_Earth 등 물리 상수.
        aux_params (dict): Cd, area_drag, mass 등 위성 제원.

    Returns:
        np.ndarray: 대기 항력에 의한 섭동 가속도 벡터 (ICRS) [m/s^2].
    """
    # 1. 대기 대비 위성의 상대 속도 벡터 계산
    # (a) ITRS 좌표계에서의 위성 속도 계산
    # v_itrs = v_icrs - omega x r_icrs (E * v_icrs 아님!)
    omega_vec = np.array([0, 0, consts['omega_Earth']])
    v_sat_itrs = E_icrs_to_itrs @ v_sat_icrs - np.cross(omega_vec, r_sat_itrs)
    
    # (b) 대기는 지구와 함께 자전하므로, 대기의 속도는 0으로 가정.
    # 따라서 위성의 상대 속도는 v_sat_itrs와 같음.
    v_rel = v_sat_itrs
    v_rel_mag = np.linalg.norm(v_rel)
    
    # 2. 항력 가속도 계산
    # B = Cd * A / m (탄도 계수)
    ballistic_coeff = aux_params['Cd'] * aux_params['area_drag'] / aux_params['mass']
    
    # a_drag = -0.5 * rho * |v_rel|^2 * B * (v_rel / |v_rel|)
    drag_force_itrs = -0.5 * density * (v_rel_mag**2) * ballistic_coeff * (v_rel / v_rel_mag)
    
    # 3. 계산된 가속도를 ICRS 좌표계로 다시 변환
    E_itrs_to_icrs = E_icrs_to_itrs.T
    drag_force_icrs = E_itrs_to_icrs @ drag_force_itrs
    
    return drag_force_icrs