import numpy as np
from datetime import timedelta
from HPOP.propagator import propagate_with_scipy


def propagate_hpop_to_surface(state_vector, start_time, force_model, max_duration_hours=12, step_seconds=60):
    """
    HPOP을 이용해 위성이 지표면에 도달할 때까지 propagate.
    step_seconds: propagate 간격(초)
    """
    import numpy as np
    from datetime import timedelta

    y = np.array(state_vector)
    if np.abs(y[:3]).max() < 10000:  # km 단위면 m로 변환
        y[:3] *= 1000
        y[3:] *= 1000

    t = start_time
    dt = timedelta(seconds=step_seconds)
    ephemeris = propagate_with_scipy(t, t+timedelta(hours=max_duration_hours), 60, y, force_model, rtol=1e-12, atol=1e-12)
    y_impact = ephemeris[-1, 1:7]
    r = y_impact[:3]
    alt = np.linalg.norm(r) / 1000.0 - 6378.137  # km
    if alt <= 0:  # 또는 100 km
        impact_time = t + dt
        impact_location = r / 1000.0  # km
        return impact_time, impact_location
    y = y_impact
       
    return None, None