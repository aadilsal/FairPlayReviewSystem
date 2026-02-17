import numpy as np
from filterpy.kalman import KalmanFilter

class BallKalmanInterpolator:
    def __init__(self):
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.x = np.array([0., 0., 0., 0.])
        
        self.kf.F = np.array([[1, 0, 1, 0],
                              [0, 1, 0, 1],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
        
        self.kf.H = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0]])
        
        self.kf.P *= 1000.
        self.kf.R = np.array([[5., 0.],
                              [0., 5.]])
        
        from filterpy.common import Q_discrete_white_noise
        self.kf.Q = Q_discrete_white_noise(dim=2, dt=1., var=0.1, block_size=2)
        
        self.initialized = False

    def reset(self, position=None):
        self.kf.x = np.array([0., 0., 0., 0.])
        self.kf.P = np.eye(4) * 500.
        self.initialized = False
        
        if position is not None:
            self.kf.x[0] = position[0]
            self.kf.x[1] = position[1]
            self.initialized = True

    def predict_next(self):
        if not self.initialized:
            return self.kf.x[:2]

        self.kf.predict() 
        return self.kf.x[:2]

    def update(self, measurement):
        if not self.initialized:
            self.reset(position=measurement)
            return self.kf.x[:2]

        self.kf.update(measurement) 
        return self.kf.x[:2]

    def get_velocity(self):
        return self.kf.x[2:]

def interpolate_trajectory(ball_infos):
    trajectory = []
    for info in ball_infos:
        if info and 'box' in info:
            trajectory.append({'position': info['box'][:2]})
        else:
            trajectory.append({'position': None})
    interpolator = BallKalmanInterpolator()
    interpolated = []
    for frame in trajectory:
        pos = frame.get('position')
        if pos is not None:
            interpolator.update(np.array(pos))
        interp = interpolator.kf.x[:2]
        interpolated.append(tuple(interp))
    return interpolated
