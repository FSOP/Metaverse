import numpy as np
from datetime import timedelta
# from astropy.time import Time
from HPOP.time_utils import convert_time_scales
from HPOP.coordinate_systems import compute_iers_matrices
from HPOP.perturbations import accel_drag
from MISC import constants as const
# from CA.RAPropagator import propagate_with_scipy
from HPOP.propagator import propagate_with_scipy
from HPOP.easyHPOP_handle import HPOP_handle


def _ballistic_propagate(
    r_icrf,
    v_icrf,
    t0,
    force_model,
    min_alt_km=0.0,
    max_time_s=1800.0,
    dt_hi=2.0,
    dt_lo=0.5,
    terminal_alt_km=5.0,
):
    # 간단한 3-DOF 점질량(중력+항력) 전파
    ephem_rows = []
    t = t0
    r = np.array(r_icrf, dtype=float)
    v = np.array(v_icrf, dtype=float)

    # MJD(UTC) 초기값
    try:
        from astropy.time import Time
        mjd_utc = Time(t).mjd
    except Exception:
        mjd_utc = force_model.aux_params.get('Mjd_UTC')

    elapsed = 0.0
    while elapsed <= max_time_s:
        alt_km = np.linalg.norm(r) / 1000.0 - 6378.137
        ephem_rows.append([t, r[0], r[1], r[2], v[0], v[1], v[2]])
        if alt_km <= min_alt_km:
            return t, r / 1000.0, np.array(ephem_rows, dtype=object)

        # 임계 고도 근처에서 하강 중이면 간단히 지면 도달 시간을 추정
        r_norm = np.linalg.norm(r)
        if r_norm > 0:
            v_r = np.dot(r, v) / r_norm  # 방사속도 (m/s)
        else:
            v_r = 0.0
        if alt_km <= terminal_alt_km and v_r < 0:
            alt_m = alt_km * 1000.0
            t_impact = alt_m / max(1.0, -v_r)
            if t_impact <= 60.0:
                t_hit = t + timedelta(seconds=t_impact)
                r_hit = r + v * t_impact
                ephem_rows.append([t_hit, r_hit[0], r_hit[1], r_hit[2], v[0], v[1], v[2]])
                return t_hit, r_hit / 1000.0, np.array(ephem_rows, dtype=object)

        # 고도에 따라 시간간격 조정 (밀도 큰 구간에서 더 촘촘하게)
        dt = dt_lo if alt_km < 30.0 else dt_hi

        # EOP 및 변환행렬 계산
        x_pole, y_pole, ut1_utc, lod, dpsi, deps, dx_pole, dy_pole, tai_utc = \
            force_model.eop_manager.get_eop_values(mjd_utc)
        mjd_ut1, mjd_tt, mjd_tdb = convert_time_scales(mjd_utc, force_model.eop_manager)
        E = compute_iers_matrices(mjd_tt, mjd_ut1, x_pole, y_pole)

        # 중력 가속도 (ICRF)
        mu = 3.986004418e14  # [m^3/s^2]
        r_norm = np.linalg.norm(r)
        a_grav = -mu * r / (r_norm**3)

        # 항력 가속도 (ICRF)
        r_itrs = E @ r
        density = force_model.atmosphere_model.get_density(mjd_utc, r_itrs)
        try:
            rho = float(np.atleast_1d(density)[0])
        except Exception:
            rho = float(density)
        a_drag = accel_drag(rho, r_itrs, v, E, force_model.aux_params)

        # 4차 Runge-Kutta로 상태 적분
        def accel(r_i, v_i, mjd_i):
            # 단계별로 밀도/항력을 재계산 (저고도에서 안정성/현실성 개선)
            r_i_norm = np.linalg.norm(r_i)
            a_grav_i = -mu * r_i / (r_i_norm**3)
            x_pole_i, y_pole_i, ut1_utc_i, lod_i, dpsi_i, deps_i, dx_pole_i, dy_pole_i, tai_utc_i = \
                force_model.eop_manager.get_eop_values(mjd_i)
            mjd_ut1_i, mjd_tt_i, mjd_tdb_i = convert_time_scales(mjd_i, force_model.eop_manager)
            E_i = compute_iers_matrices(mjd_tt_i, mjd_ut1_i, x_pole_i, y_pole_i)
            r_itrs_i = E_i @ r_i
            density_i = force_model.atmosphere_model.get_density(mjd_i, r_itrs_i)
            try:
                rho_i = float(np.atleast_1d(density_i)[0])
            except Exception:
                rho_i = float(density_i)
            a_drag_i = accel_drag(rho_i, r_itrs_i, v_i, E_i, force_model.aux_params)
            return a_grav_i + a_drag_i

        k1_r = v
        k1_v = accel(r, v, mjd_utc)

        k2_r = v + 0.5 * dt * k1_v
        k2_v = accel(r + 0.5 * dt * k1_r, v + 0.5 * dt * k1_v, mjd_utc + dt / 86400.0)

        k3_r = v + 0.5 * dt * k2_v
        k3_v = accel(r + 0.5 * dt * k2_r, v + 0.5 * dt * k2_v, mjd_utc + dt / 86400.0)

        k4_r = v + dt * k3_v
        k4_v = accel(r + dt * k3_r, v + dt * k3_v, mjd_utc + dt / 86400.0)

        r = r + (dt / 6.0) * (k1_r + 2 * k2_r + 2 * k3_r + k4_r)
        v = v + (dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)

        t = t + timedelta(seconds=dt)
        mjd_utc = mjd_utc + dt / 86400.0
        elapsed += dt

    # 제한 시간 내 지표면 도달 실패
    return None, None, np.array(ephem_rows, dtype=object)


def propagate_hpop_to_surface(state_vector, start_time, force_model, max_duration_hours=12, step_seconds=60, cutoff_alt_km=80.0, min_alt_km=0.0):
    """
    HPOP을 이용해 위성이 지표면(alt <= 0km)에 도달할 때까지 궤도 예측을 수행합니다.
    Args:
        state_vector: 초기 상태벡터 (x, y, z, vx, vy, vz), 단위 m 또는 km
        start_time: 시작 시각 (datetime)
        force_model: HPOP에서 사용할 힘 모델 객체
        max_duration_hours: 최대 propagate 시간 (시간)
        step_seconds: propagate 간격 (초)
        cutoff_alt_km: HPOP을 중단하고 탄도 전파로 전환할 고도(km)
        min_alt_km: 탄도 전파 종료 기준 고도(km)
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
    ephemeris = propagate_with_scipy(
        start_time,
        (start_time, start_time + timedelta(hours=max_duration_hours)),
        step_seconds,
        y,
        force_model,
        rtol=1e-3,
        atol=1e-3,
    )
    
    # ephemeris row format: [datetime, x, y, z, vx, vy, vz]
    # ephemeris에서 컷오프 고도 이하 최초 시점을 찾음
    cutoff_idx = None
    for i in range(ephemeris.shape[0]):
        r_i = ephemeris[i, 1:4]
        alt_i = np.linalg.norm(r_i) / 1000.0 - 6378.137  # km
        if alt_i <= cutoff_alt_km:
            cutoff_idx = i
            break

    # 컷오프 도달 시 탄도 전파로 전환
    if cutoff_idx is not None:
        t_cut = ephemeris[cutoff_idx, 0]
        r_cut = ephemeris[cutoff_idx, 1:4]
        v_cut = ephemeris[cutoff_idx, 4:7]

        impact_time, impact_location, ephem_ballistic = _ballistic_propagate(
            r_cut,
            v_cut,
            t_cut,
            force_model,
            min_alt_km=min_alt_km,
            max_time_s=1800.0,
            dt_hi=2.0,
            dt_lo=0.5,
            terminal_alt_km=5.0,
        )

        # HPOP 결과 + 탄도 결과를 이어붙임
        try:
            ephemeris_full = np.vstack([ephemeris[:cutoff_idx + 1], ephem_ballistic])
        except Exception:
            ephemeris_full = ephemeris

        return impact_time, impact_location, ephemeris_full

    # 아직 지표면 도달 못함 — 에피헤메리스는 반환하여 상위에서 샘플/업로드 가능하도록 함
    else:
        r_last = ephemeris[-1, 1:4]
        alt_last = np.linalg.norm(r_last) / 1000.0 - 6378.137
        print(f"지표면에 도달하지 않았습니다. 마지막 고도: {alt_last} km ")
        return None, None, ephemeris