import cv2
import matplotlib.pyplot as plt
# import os
#matplotlib inline

from pygame import mixer  # Load the popular external library

mixer.init()
mixer.music.load('dwayne_1.mp3')
mixer.music.play()


image = cv2.imread("dwayne_1.jpg")
height, width = image.shape[:2]
resized_image = cv2.resize(image,(3*width, 3*height), interpolation = cv2.INTER_CUBIC)

fig = plt.gcf()
fig.set_size_inches(18, 10)
plt.axis("off")
plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
plt.show()