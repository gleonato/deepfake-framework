import cv2
vidcap = cv2.VideoCapture('/home/leonato/Projects/deepfake-framework/0-DataPreparation/videos/original_sequences/youtube/c23/videos/183.mp4')
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("/home/leonato/Projects/deepfake-framework/2-FaceXray/IB/frame%d.jpg" % count, image)     # save frame as JPEG file      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1