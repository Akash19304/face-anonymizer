import cv2
import mediapipe as mp
import os
import argparse
from utils import process_img


# img_path = os.path.join('.', 'data', 'man1.jpg')
# video_path = os.path.join('.', 'data', 'testVideo.mp4')

args = argparse.ArgumentParser()  # to select the mode of input
args.add_argument("--mode", default='webcam')
args.add_argument("--filePath", default=None)

args = args.parse_args()

output_dir = './output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

try:
    mp_face_detection = mp.solutions.face_detection  # face detection

    with mp_face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=0) as face_detection:
        if args.mode in ["image"]:
            img = cv2.imread(args.filePath)
            img = process_img(img, face_detection)

            cv2.imwrite(os.path.join(output_dir, 'output.png'), img)

        elif args.mode in ["video"]:

            cap = cv2.VideoCapture(args.filePath)
            ret, frame = cap.read()

            output_video = cv2.VideoWriter(os.path.join(output_dir, 'output.mp4'),
                                           cv2.VideoWriter_fourcc(*'MP4V'),
                                           25,
                                           (frame.shape[1], frame.shape[0]))

            while ret:
                frame = process_img(frame, face_detection)
                output_video.write(frame)

                ret, frame = cap.read()

            cap.release()
            output_video.release()

        elif args.mode in ["webcam"]:
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()

            while ret:
                frame = process_img(frame, face_detection)

                cv2.imshow('frame', frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

                ret, frame = cap.read()

            cap.release()

except Exception as e:
    print("Error! ", e)



