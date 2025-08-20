import numpy as np
from RungeKutta78 import RKF78Solver

def generate_ephemeris_efficient(y0, n_step, step_size, accel_func, tolerance=1e-10):
    """
    Solver 객체를 한 번만 생성하여 더 효율적으로 작동하는 버전입니다.
    """
    total_time = n_step * step_size
    span = np.linspace(0, total_time, n_step + 1)
    
    results = np.zeros((len(span), len(y0)))
    results[0] = y0
    
    h = 0.01

    # ✨ Solver 객체를 루프 시작 전에 '한 번만' 생성합니다.
    # 초기 상태는 y0, x0=0 입니다.
    solver = RKF78Solver(
        func=accel_func,
        y0=y0,
        x0=span[0],
        h_initial=h,
        tolerance=tolerance
    )

    print("Generating Ephemeris (Efficiently)...")
    for i in range(n_step):
        # 다음 목표 시간을 설정합니다.
        target_t = span[i+1]
        
        # ✨ 이미 생성된 solver 객체의 propagate_to 메서드만 호출합니다.
        # solver는 내부적으로 자신의 상태(y, x)를 업데이트합니다.
        y_final, status = solver.propagate_to(target_t)
        
        if status != 0:
            print(f"Warning: Integration failed at step {i} with status {status}")
            break
            
        # 결과 저장
        results[i+1] = y_final
        # h 값은 solver 객체 내부에 자동으로 업데이트되므로 신경 쓸 필요가 없습니다.

    print("Done.")
    
    ephemeris_table = np.column_stack((span, results))
    return ephemeris_table


# 1. ODE 함수 정의 (예: 단진동 운동, y'' = -y)
# 상태 벡터 y는 [위치, 속도] 형태입니다. y'는 [속도, 가속도]가 됩니다.
def simple_harmonic_motion(t, y):
    position = y[0]
    velocity = y[1]
    
    d_position_dt = velocity
    d_velocity_dt = -position  # 가속도 a = -x
    
    return np.array([d_position_dt, d_velocity_dt])

# 2. 시뮬레이션 파라미터 설정
y_initial = [1.0, 0.0]  # 초기 위치: 1, 초기 속도: 0
N_STEPS = 100           # 100개의 스텝으로 나누어 결과를 봄
STEP_SIZE = 0.1         # 0.1초 간격으로 결과를 저장 (총 10초 시뮬레이션)
TOLERANCE = 1e-12       # 정밀도


# --- 사용 예시 ---
eph_efficient = generate_ephemeris_efficient(y_initial, N_STEPS, STEP_SIZE, simple_harmonic_motion, TOLERANCE)
print(eph_efficient[:10])