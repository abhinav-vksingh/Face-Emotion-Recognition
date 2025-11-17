# write_class_indices.py
import json, os
os.makedirs("models", exist_ok=True)
d = {
  "angry": 0,
  "disgust": 1,
  "fear": 2,
  "happy": 3,
  "neutral": 4,
  "sad": 5,
  "surprise": 6
}
with open("models/class_indices.json","w", encoding="utf-8") as f:
    json.dump(d, f, indent=2)
print("Wrote models/class_indices.json")
