import cv2
with open('surprise.txt','r') as f:
    img = [line.strip() for line in f]
for image in img:
    print("reading")
    loadedImage = cv2.imread("images/"+image)
    cv2.imwrite("data_set/surprise/"+image,loadedImage)
print("done writing")

