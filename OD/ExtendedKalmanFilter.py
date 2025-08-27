import sys, os
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_root)

import numpy as np
import OD.VarEqn
from datetime import datetime, timedelta
from scipy.integrate import solve_ivp
from astropy.time import Time
from OD.IOD import iod_main
from OD.coordinate_converter import coordinate_converter
from HPOP.getForceModel import getFM

from HPOP.propagator import propagate_with_scipy
# from HPOP.coordinate_systems import compute_iers_matrices

class ExtendedKalmanFilter:
    def __init__(self, initial_state, initial_covariance, obs_data, site, LTCs, Us):
        self.init_epoch = initial_state[0]
        self.Y = np.array(initial_state[1:]) * 1e3            # 상태벡터 (6,) [x, y, z, vx, vy, vz]
        self.P = initial_covariance     # 오차공분산 (6x6)

        # obs_data에서 epoch이 초기 상태보다 빠른 데이터는 제거
        obs_data_filtered = []
        LTCs_filtered = []
        Us_filtered = []
        for i in range(len(obs_data)):
            if obs_data[i, 0] > self.init_epoch:
                obs_data_filtered.append(obs_data[i])
                LTCs_filtered.append(LTCs[i])
                Us_filtered.append(Us[i])
        self.obs_data = np.array(obs_data_filtered)

        self.LTCs = LTCs_filtered
        self.Us = Us_filtered
        self.ephemeris = []             # (time, state) 저장 리스트
        self.FM = getFM()
        self.CC = coordinate_converter()
        self.site = site
        self.site_ecef = self.CC.geodetic_to_ecef(site[0], site[1], site[2])  # 관측소 위치를 ECEF로 변환
        self.sigma_ra = 0.01 * np.pi / 180  # 0.01 arcseconds to radians
        self.sigma_dec = 0.01 * np.pi / 180 # 0.01 arcseconds to radians


    def run(self):
        prev_epoch = self.init_epoch
        print("\n=== RA/Dec 관측치 vs 예측치 시간대별 비교 ===")
        print(f"{'Epoch':<25} {'RA_obs(deg)':>12} {'RA_pred(deg)':>14} {'RA_error':>12} {'Dec_obs(deg)':>14} {'Dec_pred(deg)':>14} {'Dec_error':>12}")
        for i in range(len(self.obs_data)):
            Y_old = self.Y.copy()
            LTC = self.LTCs[i]
            U = self.Us[i]
            epoch = self.obs_data[i, 0]

            # 적분 시간 계산 (초 단위)
            t = (epoch - prev_epoch).total_seconds()
            prev_epoch = epoch

            # ForceModel 시간 업데이트
            self.FM.force_model.aux_params['Mjd_UTC'] = Time(epoch).mjd

            # 상태 예측
            x_pred, Phi = self.process_model(Y_old, t, U)
            # print(f"[EKF][run] Predicted state (epoch={epoch}): {x_pred}")
            self.Y = x_pred.copy()
            self.P = self.timeupdate(self.P, Phi)

            # print(f"state({epoch}): {self.Y:.2f}")

            # 관측소 위치를 ECEF → ECI로 변환
            rs_icrf = self.CC.ecef_to_eci(self.site_ecef, epoch)
            # 관측소 기준 LOS 벡터
            los_vec = x_pred[:3] - rs_icrf
            # LOS 벡터로 RA/Dec 계산
            ra_pred, dec_pred, dRa_ds, dDec_ds = self.radec_partials(los_vec)

            # 실제 관측값 (obs_data에 radian이 아니라 degree면 np.radians 제거)
            ra_obs = self.obs_data[i, 1]
            dec_obs = self.obs_data[i, 2]

            # 오차 계산
            ra_error = ra_obs - np.degrees(ra_pred)
            dec_error = dec_obs - np.degrees(dec_pred)

            print(f"{str(epoch):<25} {ra_obs:>12.6f} {np.degrees(ra_pred):>14.6f} {ra_error:>12.6f} {dec_obs:>14.6f} {np.degrees(dec_pred):>14.6f} {dec_error:>12.6f}")

            # 관측값
            z = np.array([ra_obs, dec_obs])  # [deg, deg]
            g = np.array([np.degrees(ra_pred), np.degrees(dec_pred)])  # 예측값 [deg, deg]
            s = np.array([self.sigma_ra, self.sigma_dec]) * 180/np.pi  # 관측 노이즈 [deg, deg]
            G = np.vstack([dRa_ds, dDec_ds])  # (2, 3) → (2, 6)로 확장 필요

            # G를 6차원으로 확장 (속도에 대해 0)
            G_full = np.zeros((2, 6))
            G_full[:, :3] = G

            # 측정 업데이트
            _, self.Y, self.P = self.meas_update(self.Y, z, g, s, G_full, self.P, 6)

            ###            # 상태 예측
            x_pred, Phi = self.process_model(Y_old, t, U)

            self.Y = x_pred.copy()
            self.P = self.timeupdate(self.P, Phi)
            

            # ephemeris 저장
            self.ephemeris.append([epoch, self.Y.copy()])

        print(f"\n[EKF] Final state: {self.Y}, epoch: {self.obs_data[-1,0]}")
        return self.ephemeris

    
    def meas_update(self, x, z, g, s, G, P, n):
        """
        Extended Kalman Filter measurement update.
    
        Args:
            x (np.ndarray): State vector (n,)
            z (np.ndarray): Measurement vector (m,)
            g (np.ndarray): Predicted measurement (m,)
            s (np.ndarray): Measurement standard deviations (m,)
            G (np.ndarray): Measurement Jacobian (m, n)
            P (np.ndarray): State covariance (n, n)
    
        Returns:
            K (np.ndarray): Kalman gain (n, m)
            x_new (np.ndarray): Updated state (n,)
            P_new (np.ndarray): Updated covariance (n, n)
        """
        m = len(z)
        Inv_W = np.zeros((m, m))  # Measurement covariance (R)
        for i in range(m):
            Inv_W[i, i] = s[i] ** 2

        # G shape: (m, n)
        K = P @ G.T @ np.linalg.inv(Inv_W + G @ P @ G.T)
        x = x + K @ (z - g)
        P = (np.eye(n) - K @ G) @ P
        return K, x, P

    # 시간 업데이트
    def timeupdate(self, P, Phi):
        P_new = Phi @ P @ Phi.T
        return P_new

    def measurement_model(self, y_old, dt):
        # 관측 모델 정의
        sol = solve_ivp(self.FM.force_model, [0, dt], y_old.flatten(), method='DOP853', rtol=1e-10, atol=1e-12)
        # self.Y = sol.y[:, -1]
        return sol.y[:, -1]  # 위치 정보만 관측한다고 가정

    def process_model(self, x, dt, U):
        """
        Propagate the state vector and state transition matrix (STM, Phi) over dt seconds.
        상태벡터는 관측데이터 생성과 동일한 propagate_with_scipy로 적분,
        STM은 기존 solve_ivp(var_eqn)로 적분.
        입력값 및 적분 결과 NaN/Inf/오버플로우 방지, 진단 로그 추가.
        """
        # 입력값 체크
        x = np.asarray(x, dtype=float)  # float 배열로 강제 변환
        if dt <= 0 or np.any(np.isnan(x)) or np.any(np.isinf(x)):
            print(f"[EKF][process_model] Invalid dt or state! dt={dt}, x={x}")
            return x.flatten(), np.eye(6)

        # 현재 epoch 계산 (MJD -> datetime)
        current_mjd = self.FM.force_model.aux_params['Mjd_UTC']
        epoch_dt = Time(current_mjd, format='mjd').to_datetime()
        analysis_period = (epoch_dt, epoch_dt + timedelta(seconds=dt))

        y_initial = x.flatten()
        ephemeris = propagate_with_scipy(
            epoch_dt, analysis_period, float(dt),
            y_initial, self.FM.force_model, rtol=1e-10, atol=1e-12
        )
        if ephemeris.shape[0] == 0 or np.any(np.isnan(ephemeris)) or np.any(np.isinf(ephemeris)):
            print("[EKF][process_model] propagate_with_scipy returned empty or invalid ephemeris! Using previous state.")
            x_new = x.flatten()
        else:
            x_new = ephemeris[-1, 1:7]  # 마지막 상태벡터
            if np.any(np.isnan(x_new)) or np.any(np.isinf(x_new)): #or np.linalg.norm(x_new) > 1e6:
                print(f"[EKF][process_model] x_new has NaN/Inf/overflow! x_new={x_new}, using previous state.")
                x_new = x.flatten()

        # STM 적분 (기존 방식)
        Phi = np.eye(6)
        yPhi = np.hstack((x_new.flatten(), Phi.flatten(order='F')))
        var_eqn_obj = OD.VarEqn.VarEqn(self.FM, U)
        var_eqn = lambda t, yPhi: var_eqn_obj.var_eqn(t, yPhi)
        sol = solve_ivp(var_eqn, [0, dt], yPhi, method='DOP853', rtol=1e-10, atol=1e-12)
        yPhi_new = sol.y[:, -1]
        Phi_new = yPhi_new[6:].reshape((6, 6), order='F')

        return x_new, Phi_new

    def radec_partials(self, s):
        """
        Computes right ascension, declination, and their partial derivatives from ECI coordinates.
        Args:
            s (array-like): ECI coordinates [x, y, z]
        Returns:
            ra (float): Right Ascension [rad]
            dec (float): Declination [rad]
            dRa_ds (np.ndarray): Partials of RA w.r.t. s (3,)
            dDec_ds (np.ndarray): Partials of Dec w.r.t. s (3,)
        """
        s = np.asarray(s)
        x, y, z = s
        r_xy2 = x**2 + y**2
        r = np.linalg.norm(s)
        ra = np.arctan2(y, x)
        if ra < 0.0:
            ra += 2 * np.pi
        dec = np.arcsin(z / r)
        dRa_ds = np.array([-y / r_xy2, x / r_xy2, 0.0])
        r_xy = np.sqrt(r_xy2)
        dDec_ds = np.array([
            -x * z / (r**2 * r_xy),
            -y * z / (r**2 * r_xy),
            r_xy / r**2
        ])
        return ra, dec, dRa_ds, dDec_ds

if __name__ == "__main__":
    from MISC.DBmanager import DBmanager
    from OD.coordinate_converter import coordinate_converter
    from OD.IOD import iod_main
    import numpy as np
    from datetime import datetime

    # DB에서 관측 데이터 불러오기
    obs_data = np.array(DBmanager().get_obs_data("20250818_060440_061654"))

    # 관측소 정보 추출
    site = [float(r) for r in obs_data[0,1].split('_')]
    az_list = obs_data[:,2]
    el_list = obs_data[:,3]
    epoch_list = obs_data[:,0]

    # RA/Dec, LTC, U 변환
    cc = coordinate_converter()
    sitet = {
        'lat': site[0],
        'lon': site[1],
        'alt': site[2]
    }
    ra_list, dec_list, LTCs, Us = cc.aer_to_radec(az_list, el_list, sitet, epoch_list)

    # EKF 초기 상태 및 공분산 설정 (예시)
    # 초기 상태 추정: 첫 관측 epoch, 첫 관측 위치/속도 (실제 초기값으로 교체 필요)
    initial_epoch = epoch_list[0]

    # IOD 등을 사용하여 초기 상태 추정
    initial_state, _ = iod_main("20250818_060440_061654")
    initial_state = np.hstack((initial_state['r2_epoch'], initial_state['r2'], initial_state['v2']))
    initial_covariance = np.eye(6) * 1e-2

    # EKF 객체 생성 및 실행
    obs_data = np.hstack((obs_data[:,0].reshape(-1,1), np.array(ra_list).reshape(-1,1), np.array(dec_list).reshape(-1,1)))
    ekf = ExtendedKalmanFilter(initial_state, initial_covariance, obs_data, site, LTCs, Us)
    ephemeris = ekf.run()

    # 결과 출력
    print("최종 Ephemeris:", ephemeris)