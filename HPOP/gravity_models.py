import pyshtools as pysh
import numpy as np

class GravityModel:
    def __init__(self, model_file_path, n_max, m_max):
        print(f"중력장 모델 파일 로딩 중: {model_file_path}")
        # 4.x에서는 SHGravCoeffs.from_file() 사용
        self.coeffs = pysh.SHGravCoeffs.from_file(model_file_path, lmax=n_max, format='icgem')
        print("중력장 모델 로딩 완료.")
        self.gm = self.coeffs.gm        # 중력 상수
        self.a = self.coeffs.r0         # 기준 반경

    def compute_acceleration(self, r_itrs):
        r = np.array(r_itrs)
        # 4.x에서 직접 계산: SHGravCoeffs 객체의 'expand()'를 이용
        # 위치를 r, theta, phi로 변환
        r_mag = np.linalg.norm(r)        

        theta = np.arccos(r[2] / r_mag)       # [rad] colatitude
        phi = np.arctan2(r[1], r[0])          # [rad] longitude

        lat_deg = 90 - np.rad2deg(theta)
        lon_deg = np.rad2deg(phi)

        # expand()로 중력 퍼텐셜 계산
        g_r, g_theta, g_phi = self.coeffs.expand(lat=lat_deg, lon=lon_deg, r=r_mag)      
        

        g_x = g_r * np.sin(theta) * np.cos(phi) + \
            g_theta * np.cos(theta) * np.cos(phi) - \
            g_phi * np.sin(phi)
        g_y = g_r * np.sin(theta) * np.sin(phi) + \
            g_theta * np.cos(theta) * np.sin(phi) + \
            g_phi * np.cos(phi)
        g_z = g_r * np.cos(theta) - g_theta * np.sin(theta)

                
        return np.array([g_x, g_y, g_z])

    def get_CS(self):
        return self.coeffs.coeffs  # shape: (2, lmax+1, lmax+1)