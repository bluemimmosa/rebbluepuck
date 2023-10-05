# Python code for RED and BLUE Color Detection
import time
import RPi.GPIO as GPIO
import serial
import numpy as np
import cv2

#pin defination
redLed = 5
blueLed = 6
RBSelector = 13   

# Suppress warnings
GPIO.setwarnings(False)

# Use "GPIO" pin numbering
GPIO.setmode(GPIO.BCM)

#Use built-in internal pullup resistor so the pin is not floating
GPIO.setup(RBSelector, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(redLed, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(blueLed, GPIO.OUT, initial=GPIO.LOW)
GPIO.add_event_detect(RBSelector, GPIO.BOTH, callback=button_callback, bouncetime=100)


# Capturing video through webcam
webcam = cv2.VideoCapture(0)
webcam.set(3, 640)
webcam.set(4, 480)

#define parameters
key = 'r'
minarea = 300
start_point = (320, 0)
end_point = (320, 480)
color = (255, 255, 255)
thickness = 2
fontsize = 0.3
sizethreshold = 1000

# define mask
red_lower_1 = np.array([0, 70, 150], np.uint8) #136, 87, 111
red_upper_1 = np.array([10, 225, 255], np.uint8) #180, 255, 255
red_lower_2 = np.array([170, 70, 50], np.uint8)
red_upper_2 = np.array([180, 225, 255], np.uint8)
blue_lower = np.array([100, 50, 50], np.uint8) #94, 80, 2
blue_upper = np.array([130, 255, 255], np.uint8) #120, 255, 255

ser = serial.Serial('dev/ttyACM0', 9600, timeout=1)
ser.reset_input_buffer()

def button_callback(channel):
    if not GPIO.input(RBSelector):
        key='r'
    else:
        key='b'

# Start a while loop
while(1):



    # Reading the video from the
    # webcam in image frames
    _, imageFrame = webcam.read()

    #try blurring to see if detection improves.
    #imageFrame = cv2.GaussianBlur(imageFrame, (7, 7), 1)
  
    # Convert the imageFrame in 
    # BGR(RGB color space) to 
    # HSV(hue-saturation-value)
    # color space
    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)
    
    #draw a white line in middle
    imageFrame = cv2.line(imageFrame, start_point, end_point, color, thickness) 
  
    # Set range for red color and 
    #define mask
    prem_red_mask1 = cv2.inRange(hsvFrame, red_lower_1, red_upper_1)
    prem_red_mask2 = cv2.inRange(hsvFrame, red_lower_2, red_upper_2)
    red_mask = cv2.bitwise_or(prem_red_mask1, prem_red_mask2)
        
    # Set range for blue color and
    # define mask
    blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)
      
    # Morphological Transform, Dilation
    # for each color and bitwise_and operator
    # between imageFrame and mask determines
    # to detect only that particular color
    kernel = np.ones((5, 5), "uint8")
      
    # For red color
    red_mask = cv2.erode(red_mask, kernel)
    red_mask = cv2.dilate(red_mask, kernel)
    res_red = cv2.bitwise_and(imageFrame, imageFrame, 
                              mask = red_mask)

    # For blue color
    blue_mask = cv2.erode(blue_mask, kernel)
    blue_mask = cv2.dilate(blue_mask, kernel)
    res_blue = cv2.bitwise_and(imageFrame, imageFrame,
                               mask = blue_mask)
   
    if key == 'r':
        # Creating contour to track red color
        contours, hierarchy = cv2.findContours(red_mask,
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
      
        for pic, contour in enumerate(contours):
            approx_red = cv2.approxPolyDP(contour, 0.005*cv2.arcLength(contour, True), True)
            area = cv2.contourArea(contour)
            if(area > minarea):
                x, y, w, h = cv2.boundingRect(contour)
                cv2.drawContours(imageFrame, [approx_red], 0, (0,0,255), 2)
                #imageFrame = cv2.rectangle(imageFrame, (x, y), 
                #                           (x + w, y + h), 
                #                           (0, 0, 255), 2)
                centroid_x_red = x + int(w/2)
                distance_to_red = abs(320-centroid_x_red)
              
                cv2.putText(imageFrame, "Red Colour"+" Dist: "+str(distance_to_red)+"Area: "+str(area), (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, fontsize,
                            (0, 0, 255))
        # sorted_contours= sorted(contours, key=cv2.contourArea, reverse= True)
        # largest_item = sorted_contours[0]
        # x1, y1, w1, h1 = cv2.boundingRect(largest_item)
        # centroid1_x_red = x + int(w/2)
        # distance1_to_red = abs(320-centroid1_x_red)
        
        # ser.write("%d %d \n" %(distance1_to_red), %(cv2.contourArea(largest_item))) 
  
    if key == 'b':  
        # Creating contour to track blue color
        contours, hierarchy = cv2.findContours(blue_mask,
                                               cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)
        for pic, contour in enumerate(contours):
            approx_blue = cv2.approxPolyDP(contour, 0.005*cv2.arcLength(contour, True), True)
            area = cv2.contourArea(contour)
            if(area > minarea):
                x, y, w, h = cv2.boundingRect(contour)
                cv2.drawContours(imageFrame, [approx_blue], 0, (255,0,0), 2)
                #imageFrame = cv2.rectangle(imageFrame, (x, y),
                #                           (x + w, y + h),
                #                           (255, 0, 0), 2)
                centroid_x_blue = x + int(w/2)
                distance_to_blue = abs(320-centroid_x_blue)
                  
                cv2.putText(imageFrame, "Blue Colour"+" Dist: "+str(distance_to_blue)+"Area: "+str(area), (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            fontsize, (255, 0, 0))
    largest_contour = max(contours, key=lambda x:cv2.contourArea(x))
    x1,y1,w1,h1 = cv2.boundingRect(largest_contour)
    centroid = x1 + int(w1/2)
    distance = 320-centroid # positive = Right side, negative = Left side
    ser.write("%c %d %d \n"%key, %(distance), %(cv2.contourArea(largest_contour)))
    # if(cv2.contourArea(largest_contour) > sizethreshold):
    #     print("home")
              
    # Program Termination
    cv2.imshow("Multiple Color Detection in Real-TIme", imageFrame)
    k = cv2.waitKey(10)
    if k & 0xFF == ord('q'):
        webcam.release()
        cv2.destroyAllWindows()
        break
    elif k & 0xFF == ord('r'):
        key = 'r'
    elif k & 0xFF == ord('b'):
        key = 'b'