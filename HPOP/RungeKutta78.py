import numpy as np
import math

class RKF78Solver:
    # __init__, _step 메서드는 이전과 동일합니다.
    def __init__(self, func, y0, x0, h_initial, tolerance=1e-6):
        self.func = func
        self.y = np.array(y0, dtype=float)
        self.x = float(x0)
        self.h = float(h_initial)
        self.tolerance = float(tolerance)
        self.max_attempts = 12
        self.min_scale = 0.125
        self.max_scale = 4.0
        self.err_exponent = 1.0 / 7.0
        self.x_history = [self.x]
        self.y_history = [self.y.copy()]

    def propagate_to(self, t_target):
        """
        현재 상태(self.x)에서 목표 시간(t_target)까지 상태를 전파(적분)합니다.
        객체의 내부 상태 self.x와 self.y가 업데이트됩니다.
        """
        # 기존 solve 메서드의 로직과 완전히 동일합니다.
        if t_target < self.x: return self.y, -2
        if t_target == self.x: return self.y, 0
        
        scaled_tolerance = self.tolerance / (t_target - self.x)
        h = self.h
        
        # (이하 생략 - 기존 solve 메서드의 로직과 동일)
        # ... 루프가 끝나면 self.x, self.y, self.h가 최종 값으로 업데이트됨 ...
        
        # --- Main Integration Loop ---
        last_interval = False
        if h > (t_target - self.x):
            h = t_target - self.x
            last_interval = True
            
        while self.x < t_target:
            scale = 1.0
            
            for attempt in range(self.max_attempts):
                y_next, err_est = self._step(self.y, self.x, h)
                err_norm = np.linalg.norm(err_est)

                if err_norm == 0.0:
                    scale = self.max_scale
                    break
                
                y_norm = np.linalg.norm(self.y)
                yy = scaled_tolerance if y_norm == 0.0 else y_norm
                
                scale = 0.8 * (scaled_tolerance * yy / err_norm)**self.err_exponent
                scale = np.clip(scale, self.min_scale, self.max_scale)
                
                if err_norm < (scaled_tolerance * yy):
                    break
                
                h *= scale
                if self.x + h > t_target: h = t_target - self.x
                elif self.x + h + 0.5 * h > t_target: h = 0.5 * h
            
            if attempt >= self.max_attempts - 1:
                self.h = h * scale
                return self.y, -1
            
            self.x += h
            self.y = y_next
            h *= scale
            self.h = h
            
            self.x_history.append(self.x)
            self.y_history.append(self.y.copy())
            
            if last_interval: break
            
            if self.x + h > t_target:
                last_interval = True
                h = t_target - self.x
            elif self.x + h + 0.5 * h > t_target:
                h = 0.5 * h
                
        return self.y, 0

    def propagate(self, dt):
        """
        현재 시간(self.x)으로부터 dt만큼 상태를 전파(적분)합니다.
        """
        target_time = self.x + dt
        return self.propagate_to(target_time)

    def _step(self, y0, x0, h):
        # 이전 코드와 동일
        c_1_11=41/840; c6=34/105; c_7_8=9/35; c_9_10=9/280; a2=2/27; a3=1/9; a4=1/6; a5=5/12; a6=1/2; a7=5/6; a8=1/6; a9=2/3; a10=1/3; b31=1/36; b32=3/36; b41=1/24; b43=3/24; b51=20/48; b53=-75/48; b54=75/48; b61=1/20; b64=5/20; b65=4/20; b71=-25/108; b74=125/108; b75=-260/108; b76=250/108; b81=31/300; b85=61/225; b86=-2/9; b87=13/900; b91=2; b94=-53/6; b95=704/45; b96=-107/9; b97=67/90; b98=3; b10_1=-91/108; b10_4=23/108; b10_5=-976/135; b10_6=311/54; b10_7=-19/60; b10_8=17/6; b10_9=-1/12; b11_1=2383/4100; b11_4=-341/164; b11_5=4496/1025; b11_6=-301/82; b11_7=2133/4100; b11_8=45/82; b11_9=45/164; b11_10=18/41; b12_1=3/205; b12_6=-6/41; b12_7=-3/205; b12_8=-3/41; b12_9=3/41; b12_10=6/41; b13_1=-1777/4100; b13_4=-341/164; b13_5=4496/1025; b13_6=-289/82; b13_7=2193/4100; b13_8=51/82; b13_9=33/164; b13_10=12/41; err_factor = -41.0 / 840.0; k1 = self.func(x0, y0); k2 = self.func(x0 + a2*h, y0 + h*(a2*k1)); k3 = self.func(x0 + a3*h, y0 + h*(b31*k1 + b32*k2)); k4 = self.func(x0 + a4*h, y0 + h*(b41*k1 + b43*k3)); k5 = self.func(x0 + a5*h, y0 + h*(b51*k1 + b53*k3 + b54*k4)); k6 = self.func(x0 + a6*h, y0 + h*(b61*k1 + b64*k4 + b65*k5)); k7 = self.func(x0 + a7*h, y0 + h*(b71*k1 + b74*k4 + b75*k5 + b76*k6)); k8 = self.func(x0 + a8*h, y0 + h*(b81*k1 + b85*k5 + b86*k6 + b87*k7)); k9 = self.func(x0 + a9*h, y0 + h*(b91*k1 + b94*k4 + b95*k5 + b96*k6 + b97*k7 + b98*k8)); k10= self.func(x0 + a10*h, y0 + h*(b10_1*k1 + b10_4*k4 + b10_5*k5 + b10_6*k6 + b10_7*k7 + b10_8*k8 + b10_9*k9)); k11= self.func(x0 + h, y0 + h*(b11_1*k1 + b11_4*k4 + b11_5*k5 + b11_6*k6 + b11_7*k7 + b11_8*k8 + b11_9*k9 + b11_10*k10)); k12= self.func(x0, y0 + h*(b12_1*k1 + b12_6*k6 + b12_7*k7 + b12_8*k8 + b12_9*k9 + b12_10*k10)); k13= self.func(x0 + h, y0 + h*(b13_1*k1 + b13_4*k4 + b13_5*k5 + b13_6*k6 + b13_7*k7 + b13_8*k8 + b13_9*k9 + b13_10*k10 + k12)); y_next = y0 + h * (c_1_11*(k1 + k11) + c6*k6 + c_7_8*(k7 + k8) + c_9_10*(k9 + k10)); error_estimate = err_factor * (k1 + k11 - k12 - k13); return y_next, error_estimate

