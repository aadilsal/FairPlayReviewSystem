import numpy as np
from filterpy.kalman import KalmanFilter


def _box_center(box):
    if not box or len(box) < 4:
        return None
    x, y, w, h = box[:4]
    return (float(x + w / 2.0), float(y + h / 2.0))

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
            center = _box_center(info['box'])
            trajectory.append({'position': center})
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

def segment_aware_smooth(ball_infos, bounce_frame):
    """
    Runs Kalman smoothing in segments (e.g., pre-bounce and post-bounce).
    Resets the filter at the bounce frame to preserve the 'V' shape trajectory 
    (the DRS-critical constraint) and avoid smoothing across the bounce discontinuity.
    """
    interpolator = BallKalmanInterpolator()
    interpolated = []
    
    for i, info in enumerate(ball_infos):
        # Reset the Kalman filter completely at the exact sub-frame of bounce
        if bounce_frame is not None and i == bounce_frame:
            interpolator.reset()
            
        pos = None
        if info is not None and not info.get('ghost', False):
            if 'interpolated_position' in info and info['interpolated_position'] is not None:
                pos = info['interpolated_position']
            elif 'box' in info:
                pos = _box_center(info['box'])

        if pos is not None:
            if interpolator.initialized:
                interpolator.predict_next()
            interpolator.update(np.array(pos))
            interpolated.append((float(pos[0]), float(pos[1])))
            continue

        if interpolator.initialized:
            interpolator.predict_next()
            interp = interpolator.kf.x[:2]
            interpolated.append((float(interp[0]), float(interp[1])))
        else:
            interpolated.append(None)
        
    return interpolated
