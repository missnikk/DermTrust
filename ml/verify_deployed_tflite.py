"""
ml/verify_deployed_tflite.py
Evaluate the exact TFLite model currently residing in flutter_app/assets/models/dermtrust_classifier.tflite
on all 600 triple-verified benchmark images.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image

df = pd.read_csv("ml/data/triple_verified_600_dataset.csv")
inter = tf.lite.Interpreter(model_path="flutter_app/assets/models/dermtrust_classifier.tflite")
inter.allocate_tensors()
inp = inter.get_input_details()[0]
out = inter.get_output_details()[0]

classes = ["acne", "melasma", "psoriasis", "ringworm", "vitiligo", "not_recognised"]
corr = 0
per_cls = {c: {"tot": 0, "corr": 0, "d_tot": 0, "d_corr": 0, "l_tot": 0, "l_corr": 0} for c in classes}
fails = []

for idx, row in df.iterrows():
    arr = np.array(Image.open(row["image_path"]).convert("RGB").resize((224, 224)), dtype=np.float32) / 255.0
    inter.set_tensor(inp["index"], np.expand_dims(arr, 0))
    inter.invoke()
    probs = inter.get_tensor(out["index"])[0]
    pred = classes[int(np.argmax(probs))]
    c = row["disease"]
    tone = row["skin_tone"]
    
    per_cls[c]["tot"] += 1
    if tone == "Darker":
        per_cls[c]["d_tot"] += 1
    else:
        per_cls[c]["l_tot"] += 1
        
    if pred == c:
        corr += 1
        per_cls[c]["corr"] += 1
        if tone == "Darker":
            per_cls[c]["d_corr"] += 1
        else:
            per_cls[c]["l_corr"] += 1
    else:
        fails.append((c, pred, row["image_id"], tone))

print("=== PRODUCTION TFLITE EVALUATION ON 600 TRIPLE-VERIFIED IMAGES ===")
for c in classes:
    p = per_cls[c]
    pct = p["corr"] / p["tot"] * 100.0
    d_pct = p["d_corr"] / p["d_tot"] * 100.0
    l_pct = p["l_corr"] / p["l_tot"] * 100.0
    print(f"{c.upper():15s} | Correct: {p['corr']}/{p['tot']} ({pct:.1f}%) | Darker: {p['d_corr']}/{p['d_tot']} ({d_pct:.1f}%) | Light: {p['l_corr']}/{p['l_tot']} ({l_pct:.1f}%)")

overall_pct = corr / len(df) * 100.0
print(f"\nOVERALL ACCURACY: {corr}/{len(df)} ({overall_pct:.1f}%)")
print(f"TOTAL FAILURES: {len(fails)}")
for f in fails:
    print(f"  Failure: True [{f[0]}] predicted as [{f[1]}] (Image: {f[2]}, Tone: {f[3]})")
