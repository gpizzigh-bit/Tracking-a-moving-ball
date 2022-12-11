
import cv2 as cv
import numpy as np

def kalman_filter(cX:int, cY:int, dt: float, index: int, observation_px: list, observation_py: list):
    # initial values
    x_init = observation_px[0]
    y_init = observation_px[0]
    gravity_factor = int(((dt**2)/2) * 3661)
    vx_init = (observation_px[1]-observation_px[0])/dt
    vy_init = (observation_py[1]-observation_py[0])/dt
    init_state = [x_init, y_init, vx_init, vy_init]
    measured =[]
    measured_x = []
    measured_y = []
    predicted_px = [] 
    predicted_py = []
    mp = np.array((2,1), np.float32)
    tp = np.array((2,1), np.float32)


    param_state = 4  
    param_measured = 2

    kalman_filt = cv.KalmanFilter(param_state,param_measured)
    kalman_filt.statePost = np.array(init_state, np.float32)
    kalman_filt.statePre  = np.array(init_state, np.float32)

    # Matrix creation 
    kalman_filt.transitionMatrix = np.array([[1, 0, dt+gravity_factor, 0],
                                             [0, 1, 0, dt+gravity_factor],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], np.float32) # F matrix

    kalman_filt.measurementMatrix = np.array([[1, 0 ,0 ,0],
                                              [0, 1, 0, 0]], np.float32) # H matrix

    kalman_filt.processNoiseCov = np.array([[1, 0 ,0 ,0],
                                              [0, 1, 0, 0],
                                              [0, 0, 1, 0],
                                              [0, 0, 0, 1]],np.float32) * 1e-2 # close to zero Q matrix
    
    kalman_filt.measurementNoiseCov = np.array([[1, 0],
                                                [0, 1]], np.float32) * 1e-4 # close to zero R matrix


    kalman_filt.errorCovPost = 0.7 * np.ones((4, 4))

    # range over all the observed values until now
    # [ERROR] - Getting old values and not actual ones

    for i in range(1,index+1):
        if observation_px[i] > -1 and observation_py[i] > -1:
            mp[0] = observation_px[i] 
            mp[1] = observation_py[i]
            kalman_filt.correct(mp)
        else:
            mp[0] = 0
            mp[1] = 0

        measured.append(mp)
        measured_x.append((int(mp[0])))
        measured_y.append((int(mp[1])))

        tp = kalman_filt.predict()

        print(f"::KF:: [{index}] - x:{int(tp[0])} y: {int(tp[1])}")

        if (int(tp[0])) >= 0 and (int(tp[1])) >= 0:
            predicted_px.append((int(tp[0])))
            predicted_py.append((int(tp[1])))
        
        else:
            predicted_px.append(-1)
            predicted_py.append(-1)
    
    #print(f"::KF:: [{index}] - y: {predicted_py[-1]}")


    return len(measured), measured_x, measured_y, predicted_px, predicted_py


