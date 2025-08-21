# propagation.py

import numpy as np
from datetime import datetime
from astropy.time import Time
from scipy.integrate import solve_ivp

def altitude_event(t, y):
    r = y[:3]
    alt = np.linalg.norm(r) / 1000.0 - 6378.137  # km
    return alt  # 0이 되는 순간 이벤트 발생

altitude_event.terminal = True
# altitude_event.direction = -1  # 하강 방향에서만 이벤트 발생

def _integrate_segment(t_start_rel, t_end_rel, y0_rel, force_model, output_step_sec, rtol, atol, carry_epoch_mjd):
    """Internal helper: integrate one direction segment where y0 matches t_start_rel.

    Args:
        t_start_rel (float): segment start time (s) relative to state_epoch.
        t_end_rel (float): segment end time (s) relative to state_epoch.
        y0_rel (ndarray): state vector at t_start_rel.
        force_model (ForceModel): callable force model.
        output_step_sec (float): output step in seconds.
        rtol, atol: solver tolerances.
        carry_epoch_mjd (float): MJD of the absolute epoch corresponding to relative time 0.

    Returns:
        solution (OdeResult): raw SciPy result.
    """
    # Set force model base MJD corresponding to relative t=0, then shift by start offset
    # The ForceModel internally adds t/86400.0 to aux_params['Mjd_UTC']. We want t=0 -> current segment start.
    force_model.aux_params['Mjd_UTC'] = carry_epoch_mjd + t_start_rel/86400.0

    t_span = (0.0, t_end_rel - t_start_rel)
    total_dt = t_span[1]
    direction = 1.0 if total_dt > 0 else -1.0
    abs_dt = abs(total_dt)
    n_steps = int(round(abs_dt / float(output_step_sec)))
    if n_steps * output_step_sec != abs_dt:
        n_steps += 1
    n_steps = max(n_steps, 1)
    t_eval = np.linspace(0.0, total_dt, n_steps + 1)

    sol = solve_ivp(
        fun=force_model,
        t_span=t_span,
        y0=y0_rel,
        method='DOP853',
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        events=altitude_event
    )
    # Shift solution.t back to global relative frame
    sol.t = sol.t + t_start_rel
    return sol

def propagate_with_scipy(state_epoch, analysis_period, output_step_sec,
                         y0, force_model, rtol=1e-12, atol=1e-12):
    """
    [SciPy DOP853 사용]
    시작/종료 시각(datetime)을 입력받아 궤도를 전파하고 결과를 저장합니다.
    """
    start_time_utc = analysis_period[0]
    end_time_utc = analysis_period[1]

    # 1. 상대 시간 계산
    t_start_rel = (start_time_utc - state_epoch).total_seconds()
    t_end_rel = (end_time_utc - state_epoch).total_seconds()

    if t_start_rel == t_end_rel:
        raise ValueError("start_time_utc and end_time_utc are identical: zero-length interval")

    epoch_mjd = Time(state_epoch).mjd

    # Case A: pure forward or pure backward starting exactly at epoch
    if abs(t_start_rel) < 1e-9 or abs(t_end_rel) < 1e-9:
        if abs(t_start_rel) < 1e-9:  # start at epoch
            sol = _integrate_segment(0.0, t_end_rel, y0, force_model, output_step_sec, rtol, atol, epoch_mjd)
        else:  # end at epoch; integrate backward
            # integrate from 0 -> t_start_rel (negative). Pass t_end_rel = t_start_rel
            sol = _integrate_segment(0.0, t_start_rel, y0, force_model, output_step_sec, rtol, atol, epoch_mjd)
        # print("궤도 전파 완료.")
        ephemeris_table = np.column_stack((sol.t, sol.y.T))
        return ephemeris_table

    # Case B: interval straddles epoch (e.g., -900 to +900) while y0 at epoch
    if t_start_rel < 0.0 < t_end_rel:
        print("[info] Interval straddles state_epoch; performing two integrations (backward & forward) with y0 at epoch.")
        # Backward segment: from 0 down to t_start_rel (negative)
        sol_back = _integrate_segment(0.0, t_start_rel, y0, force_model, output_step_sec, rtol, atol, epoch_mjd)
        # Forward segment: from 0 up to t_end_rel (reuse original y0)
        sol_fwd = _integrate_segment(0.0, t_end_rel, y0, force_model, output_step_sec, rtol, atol, epoch_mjd)

        # Process altitude event termination: if either terminated early, truncate the other side after ground impact time
        # (Simplistic: if event in backward, keep; if event in forward, keep.)
        # Combine (exclude duplicated t=0 row from backward after reversing)
        t_back = sol_back.t
        y_back = sol_back.y
        # Reverse backward branch to chronological order
        t_back_rev = t_back[::-1]
        y_back_rev = y_back[:, ::-1]
        # Drop final point if duplicate of forward start (t=0)
        if abs(t_back_rev[-1]) < 1e-9:
            t_back_rev = t_back_rev[:-1]
            y_back_rev = y_back_rev[:, :-1]
        # Concatenate
        t_combined = np.concatenate([t_back_rev, sol_fwd.t])
        y_combined = np.concatenate([y_back_rev, sol_fwd.y], axis=1)
        # print("궤도 전파 완료 (양방향 결합).")
        ephemeris_table = np.column_stack((t_combined, y_combined.T))
        return ephemeris_table

    # Case C: Neither endpoint touches epoch and no straddle with y0 at epoch -> user provided y0 inconsistent
    print(f"[warn] y0 at state_epoch (0 s) but interval [{t_start_rel}, {t_end_rel}] does not include 0.\n"
          f"       Provide y0 at interval start or adjust analysis_period.")
    # Fallback: integrate from nearest endpoint that is closer to epoch and warn
    if abs(t_start_rel) < abs(t_end_rel):
        sol = _integrate_segment(t_start_rel, t_end_rel, y0, force_model, output_step_sec, rtol, atol, epoch_mjd)
    else:
        sol = _integrate_segment(t_start_rel, t_end_rel, y0, force_model, output_step_sec, rtol, atol, epoch_mjd)
    ephemeris_table = np.column_stack((sol.t, sol.y.T))
    return ephemeris_table