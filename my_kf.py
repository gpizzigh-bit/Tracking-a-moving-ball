import numpy as np


class KF:
    # x stands for the mean state vector
    # v stands for velocity 
    def __init__(self, init_x: float,
                       init_v: float,
                       accel_var: float) -> None:
        self._x = np.array([init_x, init_v]) # initial state vector
        self._P = np.eye(2) # covariance of state (in this case the identity matrix)
        self._accel_var = accel_var # variance of the acceleration
        self.H = np.array([1, 0]).reshape(1, 2)

        

    
    def predict(self, dt: float) -> None:
        # x = F x
        # P = F P Ft + G Gt a
        F = np.array([[1, dt],
                      [0, 1]])
        G = np.array([[0.5*(dt**2)],
                      [dt]])
        next_x = F @ self._x
        next_P = F @ self._P @ F.T + (G @ G.T) * self._accel_var

        self._P = next_P # update _P value
        self._x = next_x # update state _x value

    def update(self, meas_value: float, meas_variance: float) -> None:
        # y = z - H x
        # S = H P Ht + R
        # K = Pk Ht S^-1
        # so...
        # new_x = x + K y
        # new_P = (I - K H) P

        z = np.array([meas_value])
        H = np.array([1, 0]).reshape(1, 2)
        R = np.array([meas_variance])

        _y = z - H @ self._x
        S = H @ self._P @ H.T + R
        K = self._P @ H.T @ np.linalg.inv(S)

        est_x = self._x + K @ _y
        est_P = (np.identity(2) - K @ H) @ self._P

        # Update state an cov values
        self._P = est_P
        self._x = est_x

    
    @property
    def cov(self) -> np.array:
        return self._P
    
    @property
    def mean(self) -> np.array:
        return self._x
    
    @property
    def matrix_H(self) -> np.array:
        return self.H

    @property
    def pos(self) -> float:
        return self._x[0]
    
    @property
    def vel(self) -> float:
        return self._x[1]
