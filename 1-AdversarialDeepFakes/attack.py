"""
Create adversarial videos that can fool xceptionnet.

Usage:
python attack.py
    -i <folder with video files or path to video file>
    -m <path to model file>
    -o <path to output folder, will write one or multiple output videos there>

built upon the code by Andreas Rössler for detecting deep fakes.
"""

import sys, os
import argparse
from os.path import join
import cv2
import dlib
import torch
import torch.nn as nn
from PIL import Image as pil_image
from tqdm import tqdm

from network.models import model_selection
from dataset.transform import xception_default_data_transforms, mesonet_default_data_transforms
from torch import autograd
import numpy
from torchvision import transforms
import attack_algos
import json

# I don't recommend this, but I like clean terminal output.
import warnings
warnings.filterwarnings("ignore")


def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    """
    Expects a dlib face to generate a quadratic bounding box.
    :param face: dlib face class
    :param width: frame width
    :param height: frame height
    :param scale: bounding box size multiplier to get a bigger face region
    :param minsize: set minimum bounding box size
    :return: x, y, bounding_box_size in opencv form
    """
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb


def preprocess_image(image, model_type, cuda=True, legacy = False):
    """
    Preprocesses the image such that it can be fed into our network.
    During this process we envoke PIL to cast it into a PIL image.

    :param image: numpy image in opencv form (i.e., BGR and of shape
    :return: pytorch tensor of shape [1, 3, image_size, image_size], not
    necessarily casted to cuda
    """
    # Revert from BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Preprocess using the preprocessing function used during training and
    # casting it to PIL image
    if not legacy:
        # only conver to tensor here, 
        # other transforms -> resize, normalize differentiable done in predict_from_model func
        # same for meso, xception
        preprocess = xception_default_data_transforms['to_tensor']
    else:
        if model_type == "xception":
            preprocess = xception_default_data_transforms['test']
        elif model_type == "meso":
            preprocess = mesonet_default_data_transforms['test']

    preprocessed_image = preprocess(pil_image.fromarray(image))
    
    # Add first dimension as the network expects a batch
    preprocessed_image = preprocessed_image.unsqueeze(0)
    if cuda:
        preprocessed_image = preprocessed_image.cuda()

    preprocessed_image.requires_grad = True
    return preprocessed_image



def un_preprocess_image(image, size):
    """
    Tensor to PIL image and RGB to BGR
    """
    
    image.detach()
    new_image = image.squeeze(0)
    new_image = new_image.detach().cpu()

    undo_transform = transforms.Compose([
        transforms.ToPILImage(),
    ])

    new_image = undo_transform(new_image)
    new_image = numpy.array(new_image)

    new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)

    return new_image
    
def predict_with_model_legacy(image, model, model_type, post_function=nn.Softmax(dim=1),
                       cuda=True):
    """
    Predicts the label of an input image. Preprocesses the input image and
    casts it to cuda if required

    :param image: numpy image
    :param model: torch model with linear layer at the end
    :param post_function: e.g., softmax
    :param cuda: enables cuda, must be the same parameter as the model
    :return: prediction (1 = fake, 0 = real)
    """
    # Preprocess
    preprocessed_image = preprocess_image(image, model_type, cuda, legacy = True)

    # Model prediction
    output = model(preprocessed_image)
    output = post_function(output)

    # Cast to desired
    _, prediction = torch.max(output, 1)    # argmax
    prediction = float(prediction.cpu().numpy())

    return int(prediction), output


def create_adversarial_video(video_path, model_path, model_type, output_path,
                            start_frame=0, end_frame=None, attack="iterative_fgsm", 
                            compress = True, cuda=True, showlabel = True):
    """
    Reads a video and evaluates a subset of frames with the a detection network
    that takes in a full frame. Outputs are only given if a face is present
    and the face is highlighted using dlib.
    :param video_path: path to video file
    :param model_path: path to model file (should expect the full sized image)
    :param output_path: path where the output video is stored
    :param start_frame: first frame to evaluate
    :param end_frame: last frame to evaluate
    :param cuda: enable cuda
    :return:
    """
    print('Starting: {}'.format(video_path))

    # Read and write
    reader = cv2.VideoCapture(video_path)

    video_fn = video_path.split('/')[-1].split('.')[0]+'.avi'
    os.makedirs(output_path, exist_ok=True)

    if compress:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    else:
        fourcc = cv2.VideoWriter_fourcc(*'HFYU') # Chnaged to HFYU because it is lossless

    fps = reader.get(cv2.CAP_PROP_FPS)
    num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = None

    # Face detector
    face_detector = dlib.get_frontal_face_detector()

    # Load model
    if model_path is not None:
        if not cuda:
            model = torch.load(model_path, map_location = "cpu")
        else:
            model = torch.load(model_path)
        print('Model found in {}'.format(model_path))
    else:
        print('No model found, initializing random model.')
    if cuda:
        print("Converting mode to cuda")
        model = model.cuda()
        for param in model.parameters():
            param.requires_grad = True
        print("Converted to cuda")

    # raise Exception()
    # Text variables
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    font_scale = 1

    # Frame numbers and length of output video
    frame_num = 0
    assert start_frame < num_frames - 1
    end_frame = end_frame if end_frame else num_frames
    pbar = tqdm(total=end_frame-start_frame)

    metrics = {
        'total_fake_frames' : 0,
        'total_real_frames' : 0,
        'total_frames' : 0,
        'percent_fake_frames' : 0,
        'probs_list' : [],
        'attack_meta_data' : [],
    }

    while reader.isOpened():
        _, image = reader.read()
        if image is None:
            break
        frame_num += 1

        if frame_num < start_frame:
            continue
        pbar.update(1)

        # Image size
        height, width = image.shape[:2]

        # Init output writer
        if writer is None:
            writer = cv2.VideoWriter(join(output_path, video_fn), fourcc, fps,
                                     (height, width)[::-1])

            # writer = cv2.VideoWriter(join(output_path, video_fn), 0, 1,
            #                          (height, width)[::-1])

        # 2. Detect with dlib
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray, 1)
        if len(faces):
            # For now only take biggest face
            face = faces[0]

            # --- Prediction ---------------------------------------------------
            # Face crop with dlib and bounding box scale enlargement
            x, y, size = get_boundingbox(face, width, height)
            cropped_face = image[y:y+size, x:x+size]

            
            processed_image = preprocess_image(cropped_face, model_type, cuda = cuda)
            
            # Attack happening here

            # white-box attacks
            if attack == "iterative_fgsm":
                perturbed_image, attack_meta_data = attack_algos.iterative_fgsm(processed_image, model, model_type, cuda)
            elif attack == "robust":
                perturbed_image, attack_meta_data = attack_algos.robust_fgsm(processed_image, model, model_type, cuda)
            elif attack == "carlini_wagner":
                perturbed_image, attack_meta_data = attack_algos.carlini_wagner_attack(processed_image, model_type, model, cuda)

            # black-box attacks
            elif attack == "black_box":
                perturbed_image, attack_meta_data = attack_algos.black_box_attack(processed_image, model, model_type, 
                    cuda, transform_set={}, desired_acc = 0.999999)
            elif attack == "black_box_robust":
                perturbed_image, attack_meta_data = attack_algos.black_box_attack(processed_image, model, 
                    model_type, cuda, transform_set = {"gauss_blur", "translation", "resize"})
            
            # Undo the processing of xceptionnet, mesonet
            unpreprocessed_image = un_preprocess_image(perturbed_image, size)
            image[y:y+size, x:x+size] = unpreprocessed_image
            

            cropped_face = image[y:y+size, x:x+size]
            processed_image = preprocess_image(cropped_face, model_type, cuda = cuda)
            prediction, output, logits = attack_algos.predict_with_model(processed_image, model, model_type, cuda=cuda)

            print (">>>>Prediction for frame no. {}: {}".format(frame_num ,output))

            prediction, output = predict_with_model_legacy(cropped_face, model, model_type, cuda=cuda)

            print (">>>>Prediction LEGACY for frame no. {}: {}".format(frame_num ,output))

            label = 'fake' if prediction == 1 else 'real'
            if label == 'fake':
                metrics['total_fake_frames'] += 1.
            else:
                metrics['total_real_frames'] += 1.

            metrics['total_frames'] += 1.
            metrics['probs_list'].append(output[0].detach().cpu().numpy().tolist())
            metrics['attack_meta_data'].append(attack_meta_data)

            if showlabel:
                # Text and bb
                # print a bounding box in the generated video
                x = face.left()
                y = face.top()
                w = face.right() - x
                h = face.bottom() - y
                label = 'fake' if prediction == 1 else 'real'
                color = (0, 255, 0) if prediction == 0 else (0, 0, 255)
                output_list = ['{0:.2f}'.format(float(x)) for x in
                               output.detach().cpu().numpy()[0]]

                cv2.putText(image, str(output_list)+'=>'+label, (x, y+h+30),
                            font_face, font_scale,
                            color, thickness, 2)
                # draw box over face
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

        if frame_num >= end_frame:
            break

        writer.write(image)
    pbar.close()

    metrics['percent_fake_frames'] = metrics['total_fake_frames']/metrics['total_frames']

    with open(join(output_path, video_fn.replace(".avi", "_metrics_attack.json")), "w") as f:
        f.write(json.dumps(metrics))
    if writer is not None:
        writer.release()
        print('Finished! Output saved under {}'.format(output_path))
    else:
        print('Input video file was empty')


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--video_path', '-i', type=str)
    p.add_argument('--model_path', '-mi', type=str, default=None)
    p.add_argument('--model_type', '-mt', type=str, default="xception")
    p.add_argument('--output_path', '-o', type=str,
                   default='.')
    p.add_argument('--start_frame', type=int, default=0)
    p.add_argument('--end_frame', type=int, default=None)
    p.add_argument('--attack', '-a', type=str, default="iterative_fgsm")
    p.add_argument('--compress', action='store_true')
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--showlabel', action='store_true') # add face labels in the generated video

    args = p.parse_args()

    video_path = args.video_path
    if video_path.endswith('.mp4') or video_path.endswith('.avi'):
        create_adversarial_video(**vars(args))
    else:

        videos = os.listdir(video_path)
        pbar_global = tqdm(total=len(videos))
        for video in videos:
            args.video_path = join(video_path, video)
            blockPrint()
            create_adversarial_video(**vars(args))
            enablePrint()
            pbar_global.update(1)
        pbar_global.close()