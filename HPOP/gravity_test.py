# gravity_test.py (최종 수정본)

import numpy as np
import pyshtools as pysh

def calculate_gravity(r_itrs, gfc_file_path, n_max):
    """
    주어진 ITRS 좌표에서 EGM2008 모델을 사용하여 중력 가속도를 계산하는 함수.
    최신 pyshtools API 및 NumPy를 사용.
    """
    try:
        # 1. 중력 모델 계수 파일 로드
        coeffs = pysh.SHGravCoeffs.from_file(
            gfc_file_path,
            lmax=n_max,
            format='icgem'
        )

        # 2. 카테시안 -> 구면 좌표 변환
        
        r_va = np.linalg.norm(r_itrs)
        colat_rad = np.arccos(r_itrs[2] / r_va)
        lon_rad = np.arctan2(r_itrs[1], r_itrs[0])

        # 3. 구면 좌표계 기준 가속도 계산
        g_spherical = coeffs.expand(lat=colat_rad, lon=lon_rad, r=r_va)
        g_r, g_colat, g_lon = g_spherical[0], g_spherical[1], g_spherical[2]

        # ✨ 4. 구면 가속도 -> 카테시안 가속도(ax,ay,az) 직접 변환 ✨
        #    pysh.shio.sph2cart 대신 NumPy를 사용하여 직접 계산합니다.
        sin_colat = np.sin(colat_rad)
        cos_colat = np.cos(colat_rad)
        sin_lon = np.sin(lon_rad)
        cos_lon = np.cos(lon_rad)
        
        # 변환 공식을 적용하여 ax, ay, az를 계산
        ax = sin_colat * cos_lon * g_r + cos_colat * cos_lon * g_colat - sin_lon * g_lon
        ay = sin_colat * sin_lon * g_r + cos_colat * sin_lon * g_colat + cos_lon * g_lon
        az = cos_colat * g_r - sin_colat * g_colat
        
        g_cartesian = np.array([ax, ay, az])
        
        return g_cartesian

    except Exception as e:
        print(f"오류 발생: {e}")
        return None

# --- 테스트 실행 (이하 코드는 동일) ---
if __name__ == "__main__":
    gfc_file_path = './metaverse/EGM2008.gfc'
    R_EARTH = 6378137.0
    altitude = 400 * 1000.0
    test_position_itrs = np.array([0.0, 0.0, R_EARTH + altitude])
    
    print(f"테스트 위치 (ITRS): {test_position_itrs} m")
    gravity_accel = calculate_gravity(test_position_itrs, gfc_file_path, n_max=70)
    
    if gravity_accel is not None:
        magnitude = np.linalg.norm(gravity_accel)
        print("\n--- 계산 결과 ---")
        print(f"가속도 벡터 [ax, ay, az]: {gravity_accel} m/s^2")
        print(f"가속도 크기: {magnitude:.6f} m/s^2")
        print("\n테스트 성공: 가속도 크기가 약 8.68 m/s^2 근처에 나오면 정상입니다.")