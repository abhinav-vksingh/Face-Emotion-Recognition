# webcam_demo.py
"""
Webcam demo for pretrained MobileNetV2 FER model.

Usage examples (PowerShell):
  python webcam_demo.py --model ./models/fer_mobilenetv2_pretrained.h5 --classes ./models/class_indices.json
  python webcam_demo.py --model ./models/fer_mobilenetv2_pretrained.h5 --classes ./models/class_indices.json --detector haar --smooth_frames 6 --conf_thresh 0.35

Press 'q' to quit.
"""
import argparse
import json
import os
from collections import deque
import urllib.request

import cv2
import numpy as np
import tensorflow as tf

# ---------------- args ----------------
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="./models/fer_mobilenetv2_pretrained.h5")
parser.add_argument("--classes", type=str, default="./models/class_indices.json")
parser.add_argument("--camera", type=int, default=0)
parser.add_argument("--img_size", type=int, default=224)
parser.add_argument("--face_scale", type=float, default=1.2)
parser.add_argument("--smooth_frames", type=int, default=8)
parser.add_argument("--conf_thresh", type=float, default=0.40)
parser.add_argument("--consec_required", type=int, default=3)
parser.add_argument("--detector", choices=("dnn","haar"), default="dnn")
args = parser.parse_args()

# ---------------- load model + mapping ----------------
if not os.path.exists(args.model):
    raise FileNotFoundError(f"Model not found: {args.model}")
if not os.path.exists(args.classes):
    raise FileNotFoundError(f"Class indices not found: {args.classes}")

print("Loading model:", args.model)
model = tf.keras.models.load_model(args.model)
print("Model loaded. Input shape:", model.input_shape, "Output shape:", model.output_shape)

with open(args.classes, "r") as f:
    class_indices = json.load(f)
print("Raw class_indices.json:", class_indices)

def build_idx_to_class(mapping):
    idx_to_class = {}
    vals = list(mapping.values())
    # detect if values are indices (class_name->index)
    values_are_indices = len(vals) > 0 and all(str(v).isdigit() for v in vals)
    if values_are_indices:
        for cname, idx in mapping.items():
            try:
                idx_to_class[int(idx)] = str(cname)
            except:
                pass
    else:
        for k, v in mapping.items():
            try:
                idx_to_class[int(k)] = str(v)
            except:
                pass
    if len(idx_to_class) == 0:
        for k, v in mapping.items():
            try:
                idx_to_class[int(v)] = str(k)
            except:
                try:
                    idx_to_class[int(k)] = str(v)
                except:
                    pass
    return idx_to_class

idx_to_class = build_idx_to_class(class_indices)
print("Resolved idx->class mapping:", idx_to_class)

NUM_CLASSES = len(idx_to_class)
IMG_SIZE = (args.img_size, args.img_size)

# ---------------- face detector (DNN preferred) ----------------
# DNN files used by OpenCV (res10 SSD)
DNN_PROTO = "deploy.prototxt"
DNN_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"
PROTO_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
MODEL_URL = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

use_dnn = (args.detector == "dnn")
if use_dnn:
    # try to download if missing
    if not (os.path.exists(DNN_PROTO) and os.path.exists(DNN_MODEL)):
        try:
            print("DNN files missing — attempting automatic download (may take a while)...")
            if not os.path.exists(DNN_PROTO):
                urllib.request.urlretrieve(PROTO_URL, DNN_PROTO)
            if not os.path.exists(DNN_MODEL):
                urllib.request.urlretrieve(MODEL_URL, DNN_MODEL)
            print("DNN files downloaded.")
        except Exception as e:
            print("Could not download DNN files:", e)
            use_dnn = False

if use_dnn:
    net = cv2.dnn.readNetFromCaffe(DNN_PROTO, DNN_MODEL)
    print("Using DNN face detector.")
else:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    print("Using Haar cascade face detector.")

# ---------------- webcam + smoothing ----------------
cap = cv2.VideoCapture(args.camera)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

prob_buffer = deque(maxlen=args.smooth_frames)
last_shown_label = None
consec_count = 0

print("Starting webcam. Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]
    boxes = []

    if use_dnn:
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 1.0, (300,300), (104.0,177.0,123.0))
        net.setInput(blob)
        detections = net.forward()
        for i in range(detections.shape[2]):
            conf = float(detections[0,0,i,2])
            if conf > 0.5:
                box = detections[0,0,i,3:7] * np.array([w,h,w,h])
                (x0,y0,x1,y1) = box.astype("int")
                x0 = max(0, x0); y0 = max(0, y0); x1 = min(w-1, x1); y1 = min(h-1, y1)
                boxes.append((x0, y0, x1-x0, y1-y0))
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30,30))
        if len(detected) > 0:
            boxes = detected.tolist()

    for (x, y, fw, fh) in boxes:
        # enlarge
        pad_w = int((args.face_scale - 1.0) * fw / 2)
        pad_h = int((args.face_scale - 1.0) * fh / 2)
        x0 = max(0, x - pad_w); y0 = max(0, y - pad_h)
        x1 = min(w, x + fw + pad_w); y1 = min(h, y + fh + pad_h)

        roi = frame[y0:y1, x0:x1]

        try:
            face_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, IMG_SIZE)
            inp = face_resized.astype("float32") / 255.0
            inp = np.expand_dims(inp, axis=0)

            preds = model.predict(inp, verbose=0)[0]
            # align preds length vs mapping
            if len(preds) < NUM_CLASSES:
                padded = np.zeros(NUM_CLASSES, dtype=float)
                padded[:len(preds)] = preds
                preds = padded
            elif len(preds) > NUM_CLASSES:
                preds = preds[:NUM_CLASSES]

            prob_buffer.append(preds)
            avg_probs = np.mean(np.stack(prob_buffer), axis=0)

            idx = int(np.argmax(avg_probs))
            score = float(avg_probs[idx])

            if last_shown_label == idx:
                consec_count += 1
            else:
                consec_count = 1
                last_shown_label = idx

            if score >= args.conf_thresh and consec_count >= args.consec_required:
                display_label = idx_to_class.get(idx, str(idx))
            else:
                display_label = "..."

            # draw
            cv2.rectangle(frame, (x0,y0), (x1,y1), (0,255,0), 2)
            cv2.putText(frame, f"{display_label} {score:.2f}", (x0, max(15, y0-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

            # top-3 bars
            top3 = np.argsort(-avg_probs)[:3]
            base_y = y1 + 20
            for i, tid in enumerate(top3):
                cname = idx_to_class.get(int(tid), str(tid))
                tscore = float(avg_probs[tid])
                bar_w = int((x1-x0) * tscore)
                bar_y = base_y + i * 18
                cv2.rectangle(frame, (x0, bar_y), (x0+bar_w, bar_y+12), (50,150,50), -1)
                cv2.putText(frame, f"{cname}: {tscore:.2f}", (x0+2, bar_y+10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        except Exception as e:
            # on error skip this face
            pass

    cv2.imshow("FER Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
