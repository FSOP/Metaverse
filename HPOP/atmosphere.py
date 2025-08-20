# import msise00
import pymsis
import numpy as np
from datetime import datetime, timedelta

class AtmosphereModel:
    def __init__(self):
        print("NRLMSISE-00 대기 모델 초기화 완료.")

    def get_density(self, mjd_utc, r_itrs):
        # (1) MJD → datetime 변환
        mjd_epoch = datetime(1858, 11, 17, 0, 0, 0)  # MJD 기준일
        current_time = mjd_epoch + timedelta(days=mjd_utc)

        # (2) 좌표 변환 (ECEF → 위도, 경도, 고도)
        lon = np.rad2deg(np.arctan2(r_itrs[1], r_itrs[0]))
        r_xy = np.sqrt(r_itrs[0]**2 + r_itrs[1]**2)
        lat = np.rad2deg(np.arctan2(r_itrs[2], r_xy))

        a = 6378137.0  # WGS84 장반경
        b = 6356752.314245
        e_sq = (a**2 - b**2) / a**2
        N = a / np.sqrt(1 - e_sq * np.sin(np.deg2rad(lat))**2)
        alt_m = (r_xy / np.cos(np.deg2rad(lat))) - N
        alt_km = alt_m / 1000.0

        # (3) msise00 호출
        # msise00.run()
        
        result = pymsis.calculate(dates = current_time, alts=alt_km, lats=lat, lons=lon)
        
        # (4) 총 질량 밀도 [kg/m^3]
        density = result[..., pymsis.Variable.MASS_DENSITY]  # result는 dict 형태
        return density