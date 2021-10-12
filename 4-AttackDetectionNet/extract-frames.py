import cv2
import os


count = 0
attacked_dir = '/home/leonato/Projects/deepfake-framework/1-AdversarialDeepFakes/temadv/'
not_attacked_dir = '/home/leonato/Projects/deepfake-framework/0-DataPreparation/videos/manipulated_sequences/Deepfakes/raw/videos/'


# image dimension resize 
dim = (1280,720)

# # Extract frames from attacked videos

# for input_video in os.listdir(directory):
#   vidcap = cv2.VideoCapture(directory + input_video)
#   success,image = vidcap.read()
#   # count = 0
#   while success:
#     image = cv2.resize(image,dim)
#     cv2.imwrite("data/train/attacked/frame%d.jpg" % count, image)     # save frame as JPEG file      
#     success,image = vidcap.read()
#     print('Read a new frame: ', success, count)
#     count += 1

# # Extract frames from deepfaked videos (raw) 

# count = 0
# directory = '/home/leonato/Projects/deepfake-framework/0-DataPreparation/videos/manipulated_sequences/Deepfakes/raw/videos/'

for input_video in os.listdir(not_attacked_dir):
  vidcap = cv2.VideoCapture(not_attacked_dir + input_video)
  success,image = vidcap.read()
  # count = 0
  while success:
    image = cv2.resize(image,dim)
    cv2.imwrite("data/test/notattacked/frame%d.jpg" % count, image)     # save frame as JPEG file      
    success,image = vidcap.read()
    print('Read a new frame: ', success, count)
    count += 1

# Extract frames from attacked videos FOR TESTING

# for input_video in os.listdir(directory):
#   vidcap = cv2.VideoCapture(directory + input_video)
#   success,image = vidcap.read()
#   # count = 0
#   while success:
#     image = cv2.resize(image,dim)
#     cv2.imwrite("data/test/frame%d.jpg" % count, image)     # save frame as JPEG file      
#     success,image = vidcap.read()
#     print('Read a new frame: ', success, count)
#     count += 1