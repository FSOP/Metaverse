# coordinate_systems.py (pyerfa 사용 버전)

import numpy as np
# 'pysofa as sofa' 대신 'erfa'를 import 합니다.
import erfa

def compute_iers_matrices(mjd_tt, mjd_ut1, x_pole, y_pole):
    """
    pyerfa 라이브러리를 사용하여 ICRS -> ITRS 변환 행렬 E를 계산합니다.
    """
    DJM0 = 2400000.5

    # 1. 세차-장동 행렬 (sofa.pnm06a -> erfa.pnm06a)
    npb_matrix = erfa.pnm06a(DJM0, mjd_tt)

    # 2. 지구 자전 행렬
    # (a) 지구 자전 각도 (sofa.era00 -> erfa.era00)
    era = erfa.era00(DJM0, mjd_ut1)
    # (b) GAST 계산 (sofa.gst06 -> erfa.gst06)
    gast = erfa.gst06(DJM0, mjd_ut1, DJM0, mjd_tt, npb_matrix)
    # (c) Z축 기준 회전 행렬 생성 (sofa.rz -> erfa.rz)
    theta_matrix = erfa.rz(gast, np.identity(3))

    # 3. 극 운동 행렬
    # (a) TIO locator s' 계산 (sofa.sp00 -> erfa.sp00)
    sp = erfa.sp00(DJM0, mjd_tt)
    # (b) 극 운동 행렬 생성 (sofa.pom00 -> erfa.pom00)
    # arcseconds를 라디안으로 변환 (sofa.as2r -> erfa.as2r)
    x_pole_rad = np.deg2rad(x_pole/3600)
    y_pole_rad = np.deg2rad(y_pole/3600)
    pi_matrix = erfa.pom00(x_pole_rad, y_pole_rad, sp)

    # 4. 모든 행렬을 곱하여 최종 변환 행렬 E 생성
    E_matrix = pi_matrix @ theta_matrix @ npb_matrix
    
    return E_matrix