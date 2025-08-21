import numpy as np
from math import acos, atan2, sqrt
from MISC.constants import constants as const


class orcal:

    def __init__(self):
        pass

    def wrap_2pi(self, x):
        y = np.mod(x, 2*np.pi)
        return y if y >= 0.0 else y + 2*np.pi

    # -------------------------
    # r, v -> classical elements (i, Ω, e, a, ν, evec, ĥ)
    # TEME 좌표계 내 기하값이므로 궤도평면/교차선 계산에는 충분
    # -------------------------
    def elements_from_rv(self, r, v, mu=const.MU, eps_circ=1e-6):
        R = np.linalg.norm(r)
        V2 = float(np.dot(v, v))

        h = np.cross(r, v)                       # specific angular momentum
        h_norm = np.linalg.norm(h)
        h_hat = h / h_norm

        i = acos(h_hat[2])                       # inclination

        # node vector
        k = np.array([0.0, 0.0, 1.0])
        n = np.cross(k, h)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            # equatorial orbit: define RAAN = 0
            Om = 0.0
            n_hat = np.array([1.0, 0.0, 0.0])
        else:
            n_hat = n / n_norm
            Om = np.atan2(n_hat[1], n_hat[0])
            Om = self.wrap_2pi(Om)

        # eccentricity vector
        evec = (np.cross(v, h) / mu) - (r / R)
        e = np.linalg.norm(evec)

        # semi-major axis from specific mechanical energy
        a = 1.0 / (2.0 / R - V2 / mu)

        # perigee direction (for nearly circular, handle later)
        if e > eps_circ:
            p_hat = evec / e
        else:
            # near-circular: take current position projected in plane as perigee proxy
            p_hat = (r - np.dot(r, h_hat) * h_hat)
            p_norm = np.linalg.norm(p_hat)
            if p_norm < 1e-12:
                p_hat = n_hat  # fallback
            else:
                p_hat = p_hat / p_norm

        # Q-hat to form orbital plane basis (P,Q)
        q_hat = np.cross(h_hat, p_hat)
        q_hat = q_hat / np.linalg.norm(q_hat)

        # true anomaly from (P,Q) basis
        r_plane = (r - np.dot(r, h_hat) * h_hat)
        r_plane_u = r_plane / np.linalg.norm(r_plane)
        nu = np.atan2(np.dot(r_plane_u, q_hat), np.dot(r_plane_u, p_hat))
        nu = self.wrap_2pi(nu)

        return {
            'i': i, 'Om': Om, 'e': e, 'a': a, 'nu': nu,
            'h_hat': h_hat, 'p_hat': p_hat, 'q_hat': q_hat
        }

    # -------------------------
    # line of nodes between two planes (unit)
    # n_line = n1 × n2 (둘 평면 교차선 방향)
    # -------------------------
    def common_line_of_nodes(self, h1_hat, h2_hat):
        line = np.cross(h1_hat, h2_hat)
        norm = np.linalg.norm(line)
        if norm < 1e-12:
            return None
        return line / norm

    # -------------------------
    # target anomaly (asc/desc) for a given orbit when position aligns with given line (ECI)
    # - 입력 line_u: 공통 교차선 단위벡터 (ECI)
    # - asc=True  -> line_u 방향
    # - asc=False -> 반대 방향
    # 반환: ν_target ∈ [0, 2π)
    # -------------------------
    def target_true_anomaly_for_line(self, line_u, orbit, asc=True):
        # 교차선 방향을 궤도평면에 투영 후 정규화
        l_in_plane = line_u - np.dot(line_u, orbit['h_hat']) * orbit['h_hat']
        if np.linalg.norm(l_in_plane) < 1e-12:
            return None
        l_u = l_in_plane / np.linalg.norm(l_in_plane)

        # (P,Q) 기저에서의 각도
        nu_line = np.atan2(np.dot(l_u, orbit['q_hat']), np.dot(l_u, orbit['p_hat']))
        nu_line = self.wrap_2pi(nu_line)

        if asc:
            return nu_line
        else:
            return self.wrap_2pi(nu_line + np.pi)

    # -------------------------
    # convert ν -> E -> M (elliptic), then time to go from current ν to target ν
    # n = sqrt(mu/a^3), ΔM wrapped to [0, 2π)
    # -------------------------
    def time_to_reach_true_anomaly(self, orbit, nu_target):
        e = orbit['e']
        a = orbit['a']
        nu0 = orbit['nu']

        # current E, M
        def E_from_nu(nu):
            # tan(E/2) = sqrt((1-e)/(1+e)) * tan(nu/2)
            beta = np.sqrt((1 - e) / (1 + e)) if e < 1.0 else 0.0
            t = np.tan(nu/2.0)
            E = 2.0 * np.arctan2(beta * t, 1.0)
            return self.wrap_2pi(E)

        def M_from_E(E):
            return self.wrap_2pi(E - e*np.sin(E))

        E0 = E_from_nu(nu0)
        M0 = M_from_E(E0)

        Et = E_from_nu(nu_target)
        Mt = M_from_E(Et)

        dM = self.wrap_2pi(Mt - M0)

        n_rad_s = sqrt(const.MU / (a**3))  # mean motion
        if n_rad_s <= 0:
            return None

        dt = dM / n_rad_s
        return float(dt)