import numpy as np
import argparse
import time
import cv2
import os
import imutils
from imutils.video import VideoStream
from imutils.video import FPS

OUTPUT_FILE = 'output2.mp4'
if os.path.isfile(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

print("Loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
 
# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("Starting video stream...")
vs = VideoStream(src=0).start()
#width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
#height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
time.sleep(2.0)
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter(OUTPUT_FILE, fourcc, 20.0, (640, 480))
fps = FPS().start()

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
	    0.007843, (300, 300), 127.5)
 
    #get predictions by passing blob
    net.setInput(blob)
    detections = net.forward()


    for i in np.arange(0, detections.shape[2]):
	    #confidence extraction
	    confidence = detections[0, 0, i, 2]
 
	    # filter out weak detections
	    if confidence > args["confidence"]:
		    # extract the index of the class label from the
		    # `detections`, then compute the (x, y)-coordinates of
		    # the bounding box for the object
		    idx = int(detections[0, 0, i, 1])
		    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		    (startX, startY, endX, endY) = box.astype("int")
 
		    # draw the prediction on the frame
		    label = "{}: {:.2f}%".format(CLASSES[idx],
			    confidence * 100)
		    cv2.rectangle(frame, (startX, startY), (endX, endY),
			    COLORS[idx], 2)
		    y = startY - 15 if startY - 15 > 15 else startY + 15
		    cv2.putText(frame, label, (startX, y),
			    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
 
    if key == ord("q"):
        break
    out.write(frame)
    fps.update()
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
vs.stop()
cv2.destroyAllWindows()























