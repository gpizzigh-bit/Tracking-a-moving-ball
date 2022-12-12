
 ## Steps ##

# [TODO] - Import the video to the program (webcam or recording).
# [TODO] - Threshold the frames to isolate the object.
# [TODO] - Find its centroid and respective (x,y) values -> this will be our measurements. 
# [TODO] - Reproduce the video with an draw circle in this centroid position.
# [TODO] - Apply the Kalman filter to this video to detect the future movement of the object.
# [TODO] - Show tree or four prediction for the object on the video.

 # ####### #

 ## COMMENTS ##

 # --> System is slow (using old observed values) not accompanying the object and predicting its future values 
 # --> Not too precise (slightly above real value) (Wrong kalman gain?)

 # --> iam two frames behind, need to be in maximum one frame 

 #####


import cv2 as cv
import matplotlib.pyplot as plt
from find_obj import *
from kf import *



# [OK] - Import the video to the program (webcam or recording).
filename_video = "sample1.mp4"
file_path_output = "sample_1_out.mp4"
fps = get_fps(filename_video)
dt = 1/fps
scale = 2
index = 0
observation_px = []
observation_py = []
fitted_y_list = []

if __name__ == "__main__":

    cap = cv.VideoCapture(filename_video)

    # get mask
    # mask,_ = find_object_ferneback(filename_video) # this too is slow....
    mask,_ = find_object_inrange(filename_video,scale)


    # Check if camera opened successfully
    if cap.isOpened() == True: 
        # Read until video is completed
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                aux = 0
                # resize frame
                frame = cv.resize(frame, (int(frame.shape[1]/scale),int(frame.shape[0]/scale)))

                # [OK] - Threshold the frames to isolate the object.
                mask_rgb = cv.cvtColor(mask[index], cv.COLOR_GRAY2BGR)
                masked_frame = frame & mask_rgb
                
                
                # [OK] - Find its centroid and respective (x,y) values -> this will be our measurements.
                cX, cY, observation_px, observation_py = get_centroid_values(mask[index],observation_px,observation_py)
                
                if index == 31:
                    cv.imwrite("images/mask_31.jpg",mask[index][200:300, 300:400])
                    cv.imwrite("images/mask_rgb_31.jpg",masked_frame[200:300, 300:400])

                # [TODO] - Apply the Kalman filter to this video to detect the future movement of the object.

                # Wait for a minimum of 2 values
                if index >= 2:   
                    _,measured_px,measured_py,predicted_px,predicted_py = kalman_filter(cX,cY,dt,index,observation_px,observation_py)
                    if index == 31:
                        frame = show_obs_trajectory(frame,scale,cX, cY,index, observation_px, observation_py)
                        cv.imwrite("images/frame_obs_31.jpg",frame[200:300, 300:400])
                    frame = show_pred_trajectory(frame,scale, index, predicted_px, predicted_py)
                    frame = show_measured_trajectory(frame,scale,index, measured_px, measured_py)
                    print(f"::MAIN:: - [{index}](x,y) | obs : {(observation_px[-1],observation_py[-1])} | pred: {(predicted_px[-1],predicted_py[-1])} | meas : {(measured_px[-1],measured_py[-1])} ")
                    print(f"--------------------------------------------------------------------------------------------------------")
                
                # [OK] - Reproduce the video with an draw circle in this centroid position.
                frame = show_obs_trajectory(frame,scale,cX, cY,index, observation_px, observation_py)
                cv.putText(frame, f"{index}", (900,30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,100), 1)

                cv.imshow('Kalman',frame)
                index += 1 # update index

                cv.waitKey(aux)
            
                # Press P on keyboard to pause
                if cv.waitKey(25) & 0xFF == ord('n'):
                    aux = 1
                    continue

            # Break the loop on the end of video
            else:
                plt.figure(figsize=(10, 6))
                plt.gca().invert_yaxis()
                position_line = plt.plot(observation_px, observation_py, linestyle='', marker='o', color='g', label='observation')
                predicted_line = plt.plot(predicted_px, predicted_py, linestyle='', marker='o', color='b', label='prediction')
                measured_line = plt.plot(measured_px, measured_py, linestyle='', marker='x', color='y', label='measured')
                """ polifit_curve_eq = np.polynomial.polynomial.polyfit(observation_px, observation_py,2)
                poly = np.poly1d(polifit_curve_eq)
                for x in observation_px:
                    fitted_y = poly(x)
                    fitted_y_list.append(fitted_y)
                print(fitted_y_list)
                #position_line = plt.plot(observation_px,fitted_y_list, color='r', label='observation') """
                plt.legend(loc='lower right')
                plt.xlabel('x pixel')
                plt.ylabel('y pixel')
                plt.show()
                break
        # When everything done, release the video capture object
        cap.release()
        
        # Closes all the frames
        cv.destroyAllWindows()

# [TODO] - Reproduce the video with an draw circle in this centroid position.
# [TODO] - Apply the Kalman filter to this video to detect the future movement of the object.
# [TODO] - Show tree or four prediction for the object on the video.



