import numpy as np
from scipy.special import lpmv  # Associated Legendre functions

class VarEqn:
    def __init__(self, FM, U):
        self.FM = FM
        self.U = U

    def var_eqn(self, t, yPhi):
        """
        Computes the variational equations: derivative of state vector and STM.
        Args:
            t (float): Time since epoch [s]
            yPhi (np.ndarray): (42,) vector, first 6 are state, next 36 are STM (column-major)
        Returns:
            yPhip (np.ndarray): (42,) derivative vector
        """
        # --- Unpack state and STM ---
        r = yPhi[:3]         # Position vector (km)
        v = yPhi[3:6]        # Velocity vector (km/s)
        Phi = np.zeros((6, 6))
        for j in range(6):
            Phi[:, j] = yPhi[6*j+6:6*j+12]  # Column-wise

        # --- Use external force model for acceleration ---
        # 위치 벡터(r)는 km 단위이므로, force_model에 넘길 때 m 단위로 변환
        y_m = np.hstack((r * 1e3, v))  # 위치만 km->m 변환
        dY = self.FM.force_model(t, y_m)  # returns [vx, vy, vz, ax, ay, az]
        a = dY[3:6]

        # 오류 발생 시에만 경고 출력
        if np.any(np.isnan(r)) or np.any(np.isinf(r)):
            print(f"[var_eqn][ERROR] NaN/Inf in position r at t={t}, r={r}")
        if np.any(np.isnan(v)) or np.any(np.isinf(v)):
            print(f"[var_eqn][ERROR] NaN/Inf in velocity v at t={t}, v={v}")
        if np.any(np.isnan(a)) or np.any(np.isinf(a)) or np.linalg.norm(a) > 100:
            print(f"[var_eqn][ERROR] NaN/Inf or overflow in acceleration a at t={t}, a={a}, y_m={y_m}")

        # --- Compute/approximate gradient G (da/dr) ---
        G = compute_gradient_fd(r, self.U, self.FM.force_model.gravity_model, eps=1e-3)
        if np.any(np.isnan(G)) or np.any(np.isinf(G)) or np.max(np.abs(G)) > 1e-2:
            print(f"[var_eqn][ERROR] NaN/Inf or overflow in G at t={t}, r={r}, U={self.U}")
            print(f"n_max={self.FM.force_model.aux_params.get('n_max','?')}, m_max={self.FM.force_model.aux_params.get('m_max','?')}")

        # --- Build df/dy Jacobian ---
        dfdy = np.zeros((6, 6))
        for i in range(3):
            for j in range(3):
                dfdy[i, j] = 0.0                # dv/dr
                dfdy[i+3, j] = G[i, j]          # da/dr
                dfdy[i, j+3] = 1.0 if i == j else 0.0  # dv/dv
                dfdy[i+3, j+3] = 0.0            # da/dv

        # --- STM derivative ---
        Phip = dfdy @ Phi

        # --- Derivative of combined state and STM ---
        yPhip = np.zeros(42)
        yPhip[:3] = v           # dr/dt
        yPhip[3:6] = a          # dv/dt
        for i in range(6):
            for j in range(6):
                yPhip[6*j + i + 6] = Phip[i, j]  # dPhi/dt (column-major)

        # 진단: 전체 상태 NaN/Inf 체크
        if np.any(np.isnan(yPhi)) or np.any(np.isinf(yPhi)):
            print(f"[var_eqn] NaN/Inf in yPhi at t={t}, yPhi={yPhi}")
        if np.isnan(t):
            print(f"[var_eqn] nan in t!")

        return yPhip

def compute_gradient_fd(r, U, gravity_model, eps=1e-3):
    """
    Finite difference로 지구 중력장 gradient(∂a/∂r)를 계산.
    Args:
        r: (3,) inertial position
        U: (3,3) inertial->body-fixed 변환행렬
        gravity_model: GravityModel 객체 (compute_acceleration 메서드 필요)
        eps: perturbation 크기 (m)
    Returns:
        G: (3,3) gradient matrix
    """
    G = np.zeros((3, 3))
    r_m = r * 1e3  # km -> m 변환
    a0 = gravity_model.compute_acceleration(U @ r_m)
    if np.any(np.isnan(a0)) or np.any(np.isinf(a0)) or np.linalg.norm(a0) > 100:
        print(f"[compute_gradient_fd][ERROR] NaN/Inf or overflow in a0: {a0}, r_m={r_m}")
    for j in range(3):
        dr = np.zeros(3)
        dr[j] = eps
        a_plus = gravity_model.compute_acceleration(U @ (r_m + dr*1e3))
        a_minus = gravity_model.compute_acceleration(U @ (r_m - dr*1e3))
        if np.any(np.isnan(a_plus)) or np.any(np.isinf(a_plus)) or np.any(np.isnan(a_minus)) or np.any(np.isinf(a_minus)):
            print(f"[compute_gradient_fd][ERROR] NaN/Inf in a_plus/a_minus: a_plus={a_plus}, a_minus={a_minus}")
        G[:, j] = (a_plus - a_minus) / (2 * eps * 1e3)  # eps도 m 단위로 맞춤
    # G는 body-fixed 기준이므로, inertial로 변환
    if np.any(np.isnan(G)) or np.any(np.isinf(G)) or np.max(np.abs(G)) > 1e-2:
        print(f"[compute_gradient_fd][ERROR] NaN/Inf or overflow in G: G={G}, r={r}, U={U}")
    G = np.linalg.inv(U) @ G @ U
    return G
