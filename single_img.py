#
# sending sms only once per image in dataset


import cv2
import os
import cvzone
from sms import send_msg
thres = 0.55
nmsThres = 0.2

# Path to your dataset directory containing images
dataset_path = r"C:\Users\latha\Downloads\dataset"

classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().split('\n')
print(classNames)

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = "frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Get a list of image files in the dataset directory
image_files = [file for file in os.listdir(dataset_path) if file.endswith((".jpg", ".jpeg"))]

sms_sent = False  # Flag variable to track SMS sent status

for image_file in image_files:
    img = cv2.imread(os.path.join(dataset_path, image_file))
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nmsThres)

    # Send SMS only once for each image
    if not sms_sent:
        try:
            send_msg()
            sms_sent = True  # Set flag to True once SMS is sent
        except Exception as e:
            print("Error sending SMS:", e)

    for classId, conf, box in zip(classIds.flatten(), confs.flatten(), bbox):
        cvzone.cornerRect(img, box)
        cv2.putText(img, f'{classNames[classId - 1].upper()} {round(conf * 100, 2)}',
                    (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    1, (0, 255, 0), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
