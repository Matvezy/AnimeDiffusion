import cv2
import numpy as np

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min = 0, a_max = 0.999)

img =  cv2.imread("CelebA-img/0.jpg").astype(np.float)  # BGR, float
for i in range(255):
    r = np.absolute(img[:, :, 2] + 1)  # R = |R - B|
    img[:, :, 2] = np.clip(r, 0, 255)
    b = np.absolute(img[:, :, 1] - 1)  # R = |R - B|
    img[:, :, 1] = np.clip(b, 0, 255)
    g = np.absolute(img[:, :, 0] - 1)  # R = |R - B|
    img[:, :, 0] = np.clip(g, 0, 255)
    
    #cv2.imwrite('new-image.png', img)  # save the image
    cv2.imshow('img', img)
    cv2.waitKey(1)
    #img[:, :, 0] = np.absolute(img[:, :, 0] + i)
    #img[:, :, 1] = np.absolute(img[:, :, 1] + i)
img = img.astype(np.uint8)  # convert back to uint8
cv2.imshow('img', img)
cv2.waitKey()
"""
# import the necessary packages
import numpy as np
import scipy.spatial as sp
import matplotlib.pyplot as plt
import cv2

#Stored all RGB values of main colors in a array
main_colors = [(0,0,0),
                  (255,255,255),
                  (255,0,0),
                  (0,255,0),
                  (0,0,255),
                  (255,255,0),
                  (0,255,255),
                  (255,0,255),
                  ] 

image = cv2.imread("CelebA-img/0.jpg")
#convert BGR to RGB image
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

h,w,bpp = np.shape(image)

#Change colors of each pixel
#reference :https://stackoverflow.com/a/48884514/9799700
for py in range(0,h):
    for px in range(0,w):
      ########################
      #Used this part to find nearest color 
      #reference : https://stackoverflow.com/a/22478139/9799700
      input_color = (image[py][px][0],image[py][px][1],image[py][px][2])
      tree = sp.KDTree(main_colors) 
      ditsance, result = tree.query(input_color) 
      nearest_color = main_colors[result]
      ###################
      
      image[py][px][0]=nearest_color[0]
      image[py][px][1]=nearest_color[1]
      image[py][px][2]=nearest_color[2]

cv2.imshow('Ex',image)
cv2.waitKey(0) 

# show image
plt.figure()
plt.axis("off")
plt.imshow(image)
"""