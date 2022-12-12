import cv2 as cv
import numpy as np

def find_object_inrange(video: str,scale: int):

    vc = cv.VideoCapture(video)

    mask_vec = []

    # order is B ,G, R 
    lower_color1_bounds = (65,65,0)
    upper_color1_bounds = (255,255,50)

    lower_color2_bounds = (60,0,170)
    upper_color2_bounds = (120,255,210)
    
    # Read until video is completed
    while(vc.isOpened()):

        ret, frame = vc.read()
        if ret == True:

            mask1 = cv.inRange(frame,lower_color1_bounds,upper_color1_bounds)
            mask2 = cv.inRange(frame,lower_color2_bounds,upper_color2_bounds)
            mask_final = mask1 | mask2
            mask_final = cv.resize(mask_final, (int(mask_final.shape[1]/scale),int(mask_final.shape[0]/scale)))
            mask_vec.append(mask_final)
        
    # Break the loop on the end of video
        else:
            break

    return mask_vec,len(mask_vec)

# [TRY] : Calculate the Dense Optical Flow to try to find the magnitude of motion 
def find_object_ferneback(video: str):

    vc = cv.VideoCapture(video)

    flow = None
    flow_vec = []
    mask_vec = []
    img_gray_vec = []
    flow_vec = []
    flow_vec_mag = []

    if vc.isOpened() == True: 
        # get first frame
        ret, first_frame = vc.read()
        prev_frame_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
    
        # Read until video is completed
        while(vc.isOpened()):

            ret, frame = vc.read()
            if ret == True:

                new_frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                img_gray_vec.append(new_frame_gray)
                flow  = cv.calcOpticalFlowFarneback(prev_frame_gray,
                                                    new_frame_gray,
                                                    flow = flow,
                                                    pyr_scale = 0.7,
                                                    levels = 9,
                                                    winsize = 11,
                                                    iterations = 7,
                                                    poly_n = 5,
                                                    poly_sigma = 1.1,
                                                    flags = 0)

                flow_vec.append(flow)
                magnitude, angle = cv.cartToPolar(flow[...,0],flow[...,1]) # gets everyone from column 0 and 1
                flow_vec_mag.append(magnitude)
            
                # Update
                prev_frame_gray = new_frame_gray

            # Break the loop on the end of video
            else:
                break
        
        # create the binary
        for i in range(len(img_gray_vec)):
            binary_flow = (flow_vec_mag[i] > 1)*255
            binary_flow = binary_flow.astype(np.uint8)
            mask_vec.append(binary_flow)

        return mask_vec, len(mask_vec) 

def get_centroid_values(mask: classmethod, observation_px: list, observation_py: list):

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    if len(contours) <= 0: # error condition
        observation_px.append(-1)
        observation_py.append(-1) 

    #find largest contour   
    idx_max = contours[0]
    len_max = 0
    for c in contours:
        len_contour = len(c)
        if len_contour > len_max:
            len_max = len_contour
            idx_max = c
    M = cv.moments(idx_max)

    # If area of contour is zero => no detection possible
    if M['m00'] == 0 :
        observation_px.append(-1)
        observation_py.append(-1)
        cX = -1
        cY = -1
    else:
        # compute the center of the contour
        cX = int(M['m10']/M['m00'])
        cY = int(M['m01']/M['m00'])
        observation_px.append(cX)
        observation_py.append(cY)

    return cX, cY, observation_px, observation_py

def show_obs_trajectory(frame,scale: int, cX: int, cY: int, mask_index: int,observation_px: list,observation_py: list):
    #draw point with text
    cv.circle(frame, (cX, cY), scale*2, (0, 255, 0), -1)
    cv.putText(frame, "Obs", (cX - 20, cY - 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    for i in range(0,mask_index):
        cv.circle(frame, (observation_px[i],observation_py[i]),scale*2,(0, 150, 0),-1)
    
    return frame

def show_pred_trajectory(frame,scale: int, mask_index: int,predicted_px: list,predicted_py: list):
    #draw actual points with text
    cv.circle(frame, (predicted_px[-1],predicted_py[-1]), scale*2, (255, 0, 0), -1)
    cv.putText(frame, "pred", (predicted_px[-1] - 20,predicted_py[-1] - 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    # draw past points
    for i in range(0,mask_index-1):
        cv.circle(frame, (predicted_px[i],predicted_py[i]),scale*2,(150, 0, 0),-1)
    print(f"::FIND_OBJ:: [{mask_index}] - y: {predicted_py[i]}")
    
    return frame

def show_measured_trajectory(frame,scale: int, mask_index: int,measured_px: list,measured_py: list):
    #draw point with text
    cv.circle(frame, (measured_px[-1],measured_py[-1]), scale*3, (255,255,0), -1)
    cv.putText(frame, "meas", (measured_px[-1] - 20,measured_py[-1] - 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)
    for i in range(0,mask_index-1):
        cv.circle(frame, (measured_px[i],measured_py[i]),scale*3,(155,155,0),-1)
    
    return frame

def get_fps(video_name: str) -> float:

    video = cv.VideoCapture(video_name)
    (major_ver, minor_ver, subminor_ver) = (cv.__version__).split('.')

    if int(major_ver)  < 3 :
        fps = video.get(cv.cv.CV_CAP_PROP_FPS)
        return fps
    else :
        fps = video.get(cv.CAP_PROP_FPS)
        return fps

