import numpy as np
from filterpy.kalman import KalmanFilter

class BallKalmanInterpolator:
    def __init__(self):
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.x = np.array([0, 0, 0, 0]) 
        self.kf.F = np.array([[1, 0, 1, 0],
                              [0, 1, 0, 1],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0]])
        self.kf.P *= 1000.
        self.kf.R *= 5.
        self.kf.Q *= 0.01
        self.initialized = False

    def update(self, position):
        if not self.initialized:
            self.kf.x[:2] = position
            self.initialized = True
        self.kf.predict()
        self.kf.update(position)
        return self.kf.x[:2]

    def interpolate(self, trajectory):
        interpolated = []
        self.initialized = False
        for frame in trajectory:
            pos = frame.get('position')
            if pos is not None:
                interp = self.update(np.array(pos))
            else:
                interp = self.kf.x[:2]
            interpolated.append(tuple(interp))
        return interpolated

def interpolate_trajectory(ball_infos):
    trajectory = []
    for info in ball_infos:
        if info and 'box' in info:
            trajectory.append({'position': info['box'][:2]})
        else:
            trajectory.append({'position': None})
    interpolator = BallKalmanInterpolator()
    return interpolator.interpolate(trajectory)
