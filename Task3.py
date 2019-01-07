#Required imports
import cv2
import numpy as np
import math

#Reading the Image
ig1 = cv2.imread('hough.jpg')
ig2 = cv2.imread('hough.jpg')

sample = cv2.imread('hough.jpg',0)

#Methods
#Flips the kernel
def flip_operator(kernel):
    kernel_copy = [[0 for x in range(kernel.shape[1])] for y in range(kernel.shape[0])]
    #kernel_copy = kernel.copy()
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            kernel_copy[i][j] = kernel[kernel.shape[0]-i-1][kernel.shape[1]-j-1]
    kernel_copy = np.asarray(kernel_copy)
    return kernel_copy

#Convolution Logic
def convolution(image, kernel):
    #Flipping the kernel
    kernel = flip_operator(kernel)
    
    img_height = image.shape[0]
    img_width = image.shape[1]
    
    kernel_height = kernel.shape[0]
    kernel_width = kernel.shape[1]
    
    h = kernel_height//2
    w = kernel_width//2
    
    conv_result = [[0 for x in range(img_width)] for y in range(img_height)] 
      
    for i in range(h, img_height-h):
        for j in range(w, img_width-w):
            sum = 0 
            for m in range(kernel_height):
                for n in range(kernel_width):
                    sum = (sum + kernel[m][n]*image[i-h+m][j-w+n])
                    
            conv_result[i][j] = sum
    conv_result = np.asarray(conv_result)   
    return conv_result

#Defines the output image, combination of gradient_x and gradient_y
def output(img1, img2):
    h, w = img1.shape
    result = [[0 for x in range(w)] for y in range(h)] 
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            result[i][j] = (img1[i][j]**2 + img2[i][j]**2)**(1/2)
            if(result[i][j] > 255):
                result[i][j] = 255
            elif(result[i][j] < 0):
                result[i][j] =  0
    result = np.asarray(result)            
    return result

#Returns the maximum value from gradient_y/gradient_x
def maximum(gradient):
   max = gradient[0][0]
   for i in range(len(gradient)):
       for j in range(len(gradient[0])):
           if (max < gradient[i][j]):
               max = gradient[i][j]
   return max

#Returns the gradient_y/gradient_x with absolute values
def absolute_value(gradient):
    for i in range(len(gradient)):
        for j in range(len(gradient[0])):
            if(gradient[i][j] < 0):
                gradient[i][j] *= -1
            else:
                continue
    return gradient

#Plotting gradient_y
w, h = 3, 3
kernel_y = [[0 for x in range(w)] for y in range(h)] 
kernel_y = np.asarray(kernel_y)
kernel_y[0,0] = 1
kernel_y[0,1] = 2
kernel_y[0,2] = 1
kernel_y[1,0] = 0
kernel_y[1,1] = 0
kernel_y[1,2] = 0
kernel_y[2,0] = -1
kernel_y[2,1] = -2
kernel_y[2,2] = -1
gradient_y = convolution(sample, kernel_y)
gradient_y = absolute_value(gradient_y) / maximum(absolute_value(gradient_y))

#Plotting gradient_x
w, h = 3, 3
kernel_x = [[0 for x in range(w)] for y in range(h)] 
kernel_x = np.asarray(kernel_x)
kernel_x[0,0] = 1
kernel_x[0,1] = 0
kernel_x[0,2] = -1
kernel_x[1,0] = 2
kernel_x[1,1] = 0
kernel_x[1,2] = -2
kernel_x[2,0] = 1
kernel_x[2,1] = 0
kernel_x[2,2] = -1
gradient_x = convolution(sample, kernel_x)
gradient_x = absolute_value(gradient_x) / maximum(absolute_value(gradient_x))

#Plotting final output image
sobel = output(gradient_x, gradient_y)

#Thresholding on gradient_y
gradient_yy = gradient_y * 255
def check_threshold1(image):
    img_height = image.shape[0]
    img_width = image.shape[1]
    T = 19
    res = [[0 for x in range(img_width)] for y in range(img_height)]
    res = np.array(res)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if(image[i][j] > T):
                res[i][j] = 255
            else:
                res[i][j] = 0
    return res

gradient_yy = check_threshold1(gradient_yy)
cv2.imwrite("thresholded_gradient_yy.jpg",gradient_yy)

#Thresholding on gradient_x
gradient_xx = gradient_x * 255
def check_threshold2(image):
    img_height = image.shape[0]
    img_width = image.shape[1]
    T = 100
    res = [[0 for x in range(img_width)] for y in range(img_height)]
    res = np.array(res)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if(image[i][j] > T):
                res[i][j] = 255
            else:
                res[i][j] = 0
    return res
gradient_xx = check_threshold2(gradient_xx)
cv2.imwrite("thresholded_gradient_xx.jpg",gradient_xx)


def generate_accumulator(image):
  '''Reference: https://alyssaq.github.io/2014/understanding-hough-transform/'''
  thetas = np.deg2rad(np.arange(-90.0, 90.0))
  w, h = image.shape
  diag_len = int(round(math.sqrt(w*w + h*h)))  
  rhos = np.linspace(-diag_len, diag_len, diag_len * 2.0)

  cos_t = np.cos(thetas)
  sin_t = np.sin(thetas)
  num_thetas = len(thetas)

  accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint8)
  y_idxs, x_idxs = np.nonzero(image)  
  
  '''Voting'''
  for i in range(len(x_idxs)):
    x = x_idxs[i]
    y = y_idxs[i]

    for t_idx in range(num_thetas):
      rho = int(round(x * cos_t[t_idx] + y * sin_t[t_idx]) + diag_len)
      accumulator[rho, t_idx] += 1

  return accumulator, thetas, rhos

#Vertical Lines
acc1, thetas1, rhos1 = generate_accumulator(gradient_xx)
#Slanting Lines
acc2, thetas2, rhos2 = generate_accumulator(gradient_yy)

cv2.imwrite("Accumulator1.jpg", acc1)
cv2.imwrite("Accumulator2.jpg", acc2)

ls_theta1 = []
def detect_lines_slant(acc, rhos, thetas, num_iterations):
    for i in range(num_iterations):
        arr = np.unravel_index(acc.argmax(), acc.shape)
    
        acc[arr[0]-18:arr[0]+15, arr[1]-7:arr[1]+7] = 0
        rho = rhos[arr[0]]
        theta = thetas[arr[1]]
        
        a = np.cos(theta)
        b = np.sin(theta)
        
        if(i != 0):
            mean_ls_theta = np.mean(ls_theta1)
            res = np.linalg.norm(theta - mean_ls_theta)
            if(round(res, 2) <= 0.02):
                ls_theta1.append(theta)
                x0 = a*rho
                y0 = b*rho
                # these are then scaled so that the lines go off the edges of the image
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))
                
                cv2.line(ig1, (x1, y1), (x2, y2), (0, 255, 0), 3) 
        else:
            ls_theta1.append(theta)
            x0 = a*rho
            y0 = b*rho
            # these are then scaled so that the lines go off the edges of the image
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            
            cv2.line(ig1, (x1, y1), (x2, y2), (0, 255, 0), 3)
    
    cv2.imwrite("blue_lines.jpg",ig1)
    return ls_theta1

#Detecting Vertical Lines
ls_theta1 = detect_lines_slant(acc2, rhos2, thetas2, num_iterations=100)

#Detecting Slanting Lines
def detect_lines_vert(acc, rhos, thetas, num_iterations):
    for i in range(num_iterations):
        arr = np.unravel_index(acc.argmax(), acc.shape)
        acc[arr[0]-5:arr[0]+5, arr[1]-5:arr[1]+5] = 0
        rho = rhos[arr[0]]
        theta = thetas[arr[1]]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        # these are then scaled so that the lines go off the edges of the image
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
    
        cv2.line(ig2, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
    cv2.imwrite("red_lines.jpg",ig2)
    
detect_lines_vert(acc1, rhos1, thetas1, num_iterations = 6)
