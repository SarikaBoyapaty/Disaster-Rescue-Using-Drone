import cv2
import cvzone
from sms import send_msg

thres = 0.55
nmsThres = 0.2

# Path to your video file
video_path = r"C:\Users\latha\OneDrive\Desktop\WhatsApp Video 2024-04-26 at 14.44.12_e86ddb0a.mp4"

classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = "frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Get the video's frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Specify the codec and create VideoWriter object
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, (frame_width, frame_height))

sms_sent = False  # Flag variable to track SMS sent status

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    classIds, confs, bbox = net.detect(frame, confThreshold=thres, nmsThreshold=nmsThres)

    # Send SMS only once for the first frame
    if not sms_sent:
        try:
            send_msg()
            sms_sent = True  # Set flag to True once SMS is sent
        except Exception as e:
            print("Error sending SMS:", e)

    if len(classIds) != 0:
        for classId, conf, box in zip(classIds.flatten(), confs.flatten(), bbox):
            if classId == 0:  # 0 corresponds to the class "person" in COCO dataset
                cv2.rectangle(frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)
                cv2.putText(frame, f'{classNames[classId - 1].upper()} {round(conf * 100, 2)}',
                            (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            1, (0, 255, 0), 2)

    # Write the frame into the output video
    out.write(frame)

    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()