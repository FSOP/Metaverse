from datetime import datetime, timedelta, timezone
import numpy as np
import MISC.Structurer as structer

from CA.RA_ephemeris import propagate_hpop_to_surface


def _estimate_prob_crash_mc(
    state_vector,
    reentry_time,
    force_model,
    tle_epoch=None,
    n_samples=50,
    sigma_r_km0=2.0,
    sigma_v_ms0=2.0,
    growth_r_km_per_day=1.0,
    growth_v_ms_per_day=0.5,
    max_duration_hours=12,
    step_seconds=60,
):
    if state_vector is None or reentry_time is None:
        return None

    try:
        if isinstance(tle_epoch, datetime) and tle_epoch.tzinfo is None:
            tle_epoch = tle_epoch.replace(tzinfo=timezone.utc)
    except Exception:
        tle_epoch = None

    delta_days = 0.0
    try:
        if isinstance(tle_epoch, datetime) and isinstance(reentry_time, datetime):
            delta_days = abs((reentry_time - tle_epoch).total_seconds()) / 86400.0
    except Exception:
        delta_days = 0.0

    sigma_r_km = sigma_r_km0 + growth_r_km_per_day * delta_days
    sigma_v_ms = sigma_v_ms0 + growth_v_ms_per_day * delta_days

    sv = np.array(state_vector, dtype=float)
    use_km = np.max(np.abs(sv[:3])) < 10000.0
    if use_km:
        sigma_r = sigma_r_km
        sigma_v = sigma_v_ms / 1000.0
    else:
        sigma_r = sigma_r_km * 1000.0
        sigma_v = sigma_v_ms

    rng = np.random.default_rng()
    successes = 0
    total = 0
    for _ in range(int(n_samples)):
        perturb_pos = rng.normal(0.0, sigma_r, size=3)
        perturb_vel = rng.normal(0.0, sigma_v, size=3)
        sv_pert = sv.copy()
        sv_pert[:3] += perturb_pos
        sv_pert[3:] += perturb_vel
        try:
            impact_time, _, _ = propagate_hpop_to_surface(
                sv_pert.tolist(),
                reentry_time,
                force_model,
                max_duration_hours=max_duration_hours,
                step_seconds=step_seconds,
            )
            total += 1
            if impact_time is not None:
                successes += 1
        except Exception:
            continue

    if total == 0:
        return None
    return successes / total


def sample_ephemeris(ephem, step_minutes=5):
    if ephem is None:
        return ephem
    # ephem may be a numpy 2D array where each row is [datetime, y0, y1, y2, y3, y4, y5]
    try:
        seq = list(ephem)
    except Exception:
        return ephem

    # extract time stamps
    times = [row[0] for row in seq]
    if not times:
        return []
    start = times[0]
    end = times[-1]
    step = timedelta(minutes=step_minutes)
    # If less than 1 hour, keep all; else, only last 1 hour
    one_hour = timedelta(hours=1)
    t0 = max(start, end - one_hour)
    t = t0
    sampled = []
    while t <= end:
        nearest = min(seq, key=lambda p: abs((p[0] - t).total_seconds()))
        sampled.append(nearest)
        t += step
    return sampled


def process_reentry_sat(tle_or_reenter, start_epoch, ca_manager, force_model, db_man, or_cal):
    """Process single TLE for reentry analysis.

    Returns a dict matching insert_reentry_event signature or None if no event.
    """
    # support either (1) original TLE row, or (2) a dict with precomputed 'state_vector' and 'reentry_time'
    if isinstance(tle_or_reenter, dict) and 'state_vector' in tle_or_reenter and 'reentry_time' in tle_or_reenter:
        reentry_time = tle_or_reenter['reentry_time']
        state_vector = tle_or_reenter['state_vector']
    else:
        code, reentry_time, state_vector = ca_manager.get_state_at_altitude(tle_or_reenter, start_epoch, 200.0, 30, 10)
        if code != 0 or state_vector is None:
            return None

    result = propagate_hpop_to_surface(state_vector, reentry_time, force_model)
    if result is None or len(result) < 3:
        return None
    impact_time, impact_location, ephemeris = result

    # sample ephemeris to 1-minute resolution
    ephemeris_1min = sample_ephemeris(ephemeris, 1)
    ephem1 = structer.reassemble_orbit(ephemeris_1min)

    # determine norad id depending on input type
    if isinstance(tle_or_reenter, dict):
        norad = tle_or_reenter.get('norad')
    else:
        norad = tle_or_reenter[0]
    sat_info = db_man.get_SATCAT_info(norad)

    inst_eop = force_model.get_eop()
    inst_eop['Date'] = start_epoch.strftime('%Y%m%d')
    inbound_info = structer.inbound_info(or_cal.cal_inbound(ephemeris_1min), inst_eop)

    creation_date = datetime.now(timezone.utc)
    start_time = reentry_time
    # impact_time이 None일 수 있으므로 안전하게 crash_id 생성
    if impact_time is not None:
        crash_id = f"{impact_time.strftime('%Y%m%d_%H%M%S')}_{norad}"
    else:
        crash_id = f"noimpact_{norad}_{creation_date.strftime('%Y%m%d_%H%M%S')}"
    norad_cat_id = sat_info['NORAD_CAT_ID']
    object_name = sat_info['OBJECT_NAME']
    object_type = sat_info['OBJECT_TYPE']
    rcs = sat_info['RCS']

    orbit_crash = repr(ephem1)
    time_crash = impact_time
    lat_crash = lon_crash = alt_crash = None
    if impact_location is not None and len(impact_location) >= 3:
        lat_crash, lon_crash, alt_crash = impact_location[0], impact_location[1], impact_location[2]

    # 확률 추정 (TLE epoch 기준 시간 경과에 따른 오차 모델을 가정한 몬테카를로)
    tle_epoch = tle_or_reenter.get('tle_epoch') if isinstance(tle_or_reenter, dict) else None
    # 개발 단계: 샘플 수를 줄여 빠르게 실행 (권장: 50~200)
    prob_crash = _estimate_prob_crash_mc(
        state_vector,
        reentry_time,
        force_model,
        tle_epoch=tle_epoch,
        n_samples=10,
        sigma_r_km0=2.0,
        sigma_v_ms0=2.0,
        growth_r_km_per_day=1.0,
        growth_v_ms_per_day=0.5,
        max_duration_hours=12,
        step_seconds=60,
    )
    inbound_info_str = repr(inbound_info)

    return {
        'crash_id': crash_id,
        'creation_date': creation_date,
        'start_time': start_time,
        'norad_cat_id': norad_cat_id,
        'object_name': object_name,
        'object_type': object_type,
        'rcs': rcs,
        'orbit_crash': orbit_crash,
        'time_crash': time_crash,
        'lat_crash': lat_crash,
        'lon_crash': lon_crash,
        'alt_crash': alt_crash,
        'prob_crash': prob_crash,
        'inbound_info_str': inbound_info_str,
        'ephemeris_sampled': ephemeris_1min,
        'SAT_struct': structer.SAT_struc(norad_cat_id, object_name, object_type, rcs, ephem1, 0, "")
    }
