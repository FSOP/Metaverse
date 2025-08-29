# force_models.py

import numpy as np
from HPOP.time_utils import convert_time_scales
from HPOP.coordinate_systems import compute_iers_matrices
from HPOP.perturbations import accel_point_mass, accel_solar_radiation_pressure, accel_drag
from MISC import constants as const

# --------------------------------------------------------------------------
# 세부 계산을 위한 빈 함수들 (나중에 채워나갈 부분)
# --------------------------------------------------------------------------

def get_time_variables(mjd_utc, ut1_utc, tai_utc):
    """다양한 시간 척도(UT1, TT, TDB 등)를 계산합니다."""
    print("    (TODO: 다양한 시간 척도 계산...)")
    # 임시 반환 값
    return mjd_utc, mjd_utc, mjd_utc # MJD_UT1, MJD_TT, MJD_TDB

def get_planetary_ephemeris(mjd_tdb):
    """JPL DE440 천체력을 이용해 행성, 태양, 달의 위치를 계산합니다."""
    print(f"    (TODO: {mjd_tdb:.2f} 시점의 행성 위치 계산...)")
    # 임시 반환 값 (3차원 벡터)
    return {
        'sun': np.zeros(3), 'moon': np.zeros(3), 'mercury': np.zeros(3),
        'venus': np.zeros(3), 'earth': np.zeros(3), 'mars': np.zeros(3),
        'jupiter': np.zeros(3), 'saturn': np.zeros(3), 'uranus': np.zeros(3),
        'neptune': np.zeros(3), 'pluto': np.zeros(3), 'sun_ssb': np.zeros(3)
    }

def accel_relativity(r_sat, v_sat):
    """상대론적 효과에 의한 가속도를 계산합니다."""
    return np.zeros(3)


# --------------------------------------------------------------------------
# Accel 함수의 역할을 하는 메인 클래스
# --------------------------------------------------------------------------
class ForceModel:
    """
    위성에 가해지는 모든 섭동힘을 계산하는 모델 클래스입니다.
    MATLAB의 Accel.m 함수의 구조와 역할을 따릅니다.
    """
    def __init__(self, aux_params, eop_manager, ephem_manager, gravity_model, atmosphere_model):
        """
        클래스 생성 시 시뮬레이션에 필요한 모든 고정 데이터와 설정을 저장합니다.
        
        Args:
            aux_params (dict): 위성 제원(질량, 면적) 및 어떤 힘을 켤지 결정하는 플래그.
            eop_data: IERS 지구 방향 매개변수 데이터.
        """
        self.aux_params = aux_params
        self.eop_manager = eop_manager
        self.ephem_manager = ephem_manager
        self.gravity_model = gravity_model # GravityModel 객체 저장
        self.atmosphere_model = atmosphere_model
        print("ForceModel 객체가 모든 관리자와 함께 생성되었습니다.")
            
            
    def __call__(self, t, y):
        """
        주어진 시간(t)과 상태(y)에 대한 가속도를 계산합니다.
        이 메서드 덕분에 ForceModel 객체를 함수처럼 적분기에 전달할 수 있습니다.
        
        Args:
            t (float): 기준 시점(Mjd_UTC)으로부터 흐른 시간 (초).
            y (np.ndarray): 위성의 상태 벡터 [x, y, z, vx, vy, vz] (ICRF).
        
        Returns:
            np.ndarray: 상태 벡터의 미분 [vx, vy, vz, ax, ay, az].
        """
        # --- 1. 시간 변수 계산 ---
        mjd_utc = self.aux_params['Mjd_UTC'] + t / 86400.0
        # mjd_utc = self.aux_params['Mjd_UTC'] 
        
        # IERS 데이터 조회 (EOP)
        x_pole, y_pole, ut1_utc, lod, dpsi, deps, dx_pole, dy_pole, tai_utc = \
            self.eop_manager.get_eop_values(mjd_utc)

        # 다양한 시간 척도로 변환
        mjd_ut1, mjd_tt, mjd_tdb = convert_time_scales(mjd_utc, self.eop_manager)
        
        # --- 2. 좌표계 변환 행렬 계산 ---
        E = compute_iers_matrices(mjd_tt, mjd_ut1, x_pole, y_pole)
        E_transpose = E.T # ITRS -> ICRS

        # --- 3. 행성, 태양, 달 위치 계산 ---
        # EphemerisManager 객체를 사용하여 위치 정보를 가져옵니다.   
        # EphemerisManager는 위치를 km 단위로 반환하므로 m 단위로 변환합니다.
        ephem_km = self.ephem_manager.get_positions(mjd_tdb)
        ephem_m = {name: pos * 1000.0 for name, pos in ephem_km.items()}

        # --- 4. 가속도 계산 ---
        # 모든 가속도를 더해나갈 3차원 벡터 초기화
        # --- 4. 가속도 계산 ---
        total_acceleration = np.zeros(3)
        r_sat_icrs = y[:3] # [m] 위성 위치 (ICRS)
        v_sat_icrs = y[3:6] # [m/s]        
        
        # (a) 지구 중력장 가속도
        # 위치를 ICRS -> ITRS로 변환하여 GravityModel에 전달
        r_sat_itrs = E @ r_sat_icrs
        v_sat_itrs = E @ v_sat_icrs

        accel_itrs = self.gravity_model.compute_acceleration(r_sat_itrs)
        
        # accel_icrs 변수를 다시 사용하여 계산 결과를 저장합니다.
        accel_harmonic_icrs = E_transpose @ accel_itrs
        total_acceleration += accel_harmonic_icrs
        
        # (b) 제3체 섭동 (달, 태양)
        r_earth_ssb_m = ephem_m['earth']
        accel_sun = np.zeros(3)
        if self.aux_params.get('sun', False):
            accel_sun = accel_point_mass(r_sat_icrs, r_earth_ssb_m, ephem_m['sun'], const.GM_Sun)
            total_acceleration += accel_sun

        accel_moon = np.zeros(3)
        if self.aux_params.get('moon', False):
            accel_moon = accel_point_mass(r_sat_icrs, r_earth_ssb_m, ephem_m['moon'], const.GM_Moon)
            total_acceleration += accel_moon

        # (c) 태양복사압 (SRP)
        accel_srp = np.zeros(3)
        if self.aux_params.get('sRad', False):
            accel_srp = accel_solar_radiation_pressure(
                r_sat_icrs, 
                ephem_m['earth'], 
                ephem_m['sun'],
                self.aux_params
            )
            total_acceleration += accel_srp

        # (d) 대기항력
        accel_ad = np.zeros(3)
        if self.aux_params.get('drag', False):
            density = self.atmosphere_model.get_density(mjd_utc, r_sat_itrs)
            accel_ad = accel_drag(density, r_sat_itrs, v_sat_icrs, E, self.aux_params)
            total_acceleration += accel_ad
            
        # # (f) 상대론적 효과
        # if self.aux_params.get('Relativity', False):
        #     total_acceleration += accel_relativity(r_sat, v_sat)
            
        # --- 5. 최종 결과 반환 ---
        # dY/dt = [속도; 가속도]
        dY = np.hstack((v_sat_icrs, total_acceleration))
        
        return dY
    
    def get_eop(self):
        x_pole, y_pole, ut1_utc, lod, dpsi, deps, dx_pole, dy_pole, tai_utc = self.eop_manager.get_eop_values(self.aux_params['Mjd_UTC'])
        return {
            'date': self.aux_params['Mjd_UTC'],
            'x_pole': x_pole,
            'y_pole': y_pole,
            'ut1_utc': ut1_utc,
            'lod': lod,
            'dpsi': dpsi,
            'deps': deps,
            'dx_pole': dx_pole,
            'dy_pole': dy_pole,
            'DAT': tai_utc,
            'DATA_TYPE': 'P'
        }