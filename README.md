# Infant Distress Classification

Code and pretrained models for **real-world infant distress detection** from home (LENA-style) audio.

Paper (accepted, Interspeech 2026): **Advancing Infant Distress Detection: Two- and Three-Way Classification in Real-World Audio Environments** (Galhotra, Khante, Madden-Rusnak, de Barbaro).

Pipeline: **5-second windows with 4-second overlap** (1-second hop) → **YAMNet embeddings → PCA → RBF SVM** → majority vote to **per-second** labels.

Dataset: [deBarbaroFussCry on HomeBank](https://talkbank.org/homebank/access/Password/deBarbaroFussCry.html)  
Code: [github.com/dailyactivitylab/InfantDistressClassification](https://github.com/dailyactivitylab/InfantDistressClassification)

---

## What is in this repo

| Path | Role |
|------|------|
| [`weights/binary/`](weights/binary) | **2-way weights.** `0` = non-distress, `1` = distress (fuss and cry already merged). |
| [`weights/ternary/`](weights/ternary) | **3-way weights.** `0` = non-distress, `1` = fuss, `2` = cry. |
| [`predict.py`](predict.py) | Run either model on a WAV file. |
| [`fine_tune_binary.py`](fine_tune_binary.py) | Adapt the 2-way model on your labeled clips. |
| [`fine_tune_ternary.py`](fine_tune_ternary.py) | Adapt the 3-way model on your labeled clips. |
| [`examples/test.wav`](examples/test.wav) | Short sample clip. |
| [`requirements.txt`](requirements.txt) | Python 3.11 packages for predict and fine-tune. |

Each weights folder has the same three files: `svm_model.pkl`, `scaler.pkl`, `pca.pkl`. Point `--model_dir` at the folder, not at a single `.pkl`.

---

## Binary vs ternary

| | `weights/binary` (2-way) | `weights/ternary` (3-way) |
|--|--------------------------|---------------------------|
| `0` | non-distress | non-distress |
| `1` | **distress** (fuss **and** cry together) | **fuss** |
| `2` | not used | **cry** |

Use **binary** when you only need distress vs not. Use **ternary** when you need fuss and cry kept apart.

The paper trains a dedicated binary SVM (fuss+cry merged before training). If you only have the ternary model and want a 2-way label, you can merge after the fact:

```python
distress = 0 if pred == 0 else 1   # 1 and 2 → distress
```

That merge is a convenience, not a replay of the paper’s binary SVM.

---

## Setup

Use **Python 3.11** (TensorFlow is unreliable on older system Pythons on Apple Silicon).

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Audio: `.wav`, ideally 16 kHz, at least **5 seconds**.

---

## Predict

```bash
# 2-way: 0 = non-distress, 1 = distress
python predict.py --audio examples/test.wav --mode binary

# 3-way: 0 = non-distress, 1 = fuss, 2 = cry
python predict.py --audio examples/test.wav --mode ternary
```

`--model_dir` defaults to `weights/binary` or `weights/ternary` from `--mode`. Override if you trained or fine-tuned your own copy:

```bash
python predict.py --audio path/to/file.wav --mode ternary --model_dir ./finetuned_ternary
```

Or in Python (same windowing as the paper: 5 s + 4 s overlap):

```python
import joblib, librosa, numpy as np, tensorflow as tf, tensorflow_hub as hub
from collections import Counter

SAMPLE_RATE, CHUNK, OVERLAP = 16000, 5 * 16000, 4 * 16000

def chunks(audio):
    out, i = [], 0
    while i + CHUNK <= len(audio):
        out.append(audio[i:i + CHUNK]); i += CHUNK - OVERLAP
    if i < len(audio) and len(audio) - i >= SAMPLE_RATE:
        out.append(audio[i:])
    return out

def embed(xs, yamnet):
    embs = []
    for x in xs:
        w = tf.convert_to_tensor(x, tf.float32)
        w = tf.concat([w, tf.zeros([max(0, SAMPLE_RATE - tf.shape(w)[0])], tf.float32)], 0)[:SAMPLE_RATE]
        _, e, _ = yamnet(tf.reshape(w, [-1]))
        embs.append(e.numpy()[0])
    return np.array(embs)

def vote(preds):
    bucket = {}
    for i, p in enumerate(preds):
        for s in range(i, i + 5):
            bucket.setdefault(s, []).append(int(p))
    return np.array([Counter(bucket[s]).most_common(1)[0][0] for s in range(max(bucket) + 1)])

svm = joblib.load("weights/ternary/svm_model.pkl")
scaler = joblib.load("weights/ternary/scaler.pkl")
pca = joblib.load("weights/ternary/pca.pkl")
yamnet = hub.KerasLayer("https://tfhub.dev/google/yamnet/1")
audio, _ = librosa.load("examples/test.wav", sr=SAMPLE_RATE)
pred = vote(svm.predict(pca.transform(scaler.transform(embed(chunks(audio), yamnet)))))
```

---

## Fine-tune on your data

Labels are headerless CSVs: `start,end,label`  
`label` may be `0/1/2` or `fuss`/`cry`. Files are paired by **filename** (`P12_1.wav` ↔ `P12_1.csv`), even if folder names differ.

```text
audio/P12/P12_1.wav
labels/P12/P12_1.csv
```

### 2-way (distress vs not)

`fine_tune_binary.py` maps **fuss and cry → 1**. Numeric `2` is treated as distress, not as a third class.

```bash
python fine_tune_binary.py \
  --model_path weights/binary \
  --data_folder /path/to/audio \
  --label_folder /path/to/labels \
  --output_path ./finetuned_binary
```

### 3-way (non-distress / fuss / cry)

```bash
python fine_tune_ternary.py \
  --model_path weights/ternary \
  --data_folder /path/to/audio \
  --label_folder /path/to/labels \
  --output_path ./finetuned_ternary
```

Both scripts reuse the released **scaler + PCA** and train a **new RBF SVM** on your clips (undersampled). Outputs are drop-in `svm_model.pkl`, `scaler.pkl`, `pca.pkl`.

---

## Method (paper)

- Compared 1 s, 5 s, and **5 s + 4 s overlap**; overlapping 5 s windows were used for all experiments.
- Class imbalance: **random undersampling**.
- Evaluation in the paper: **leave-one-participant-out (LOPO-CV)**.
- Features: YAMNet embeddings, **PCA (~90% variance)**, **RBF SVM**.
- Ternary paper result (YAMNet + RBF SVM, macro-averaged): precision 0.602, recall 0.656, **F1 0.624**.

---

## Desktop app

A local UI is in [Releases](https://github.com/dailyactivitylab/InfantDistressClassification/releases) (`InfantDistressDetector.zip` / `InfantDistressDetector_Simple.exe`). It runs at `http://localhost:5005` and stays on your machine. The current app is the **2-way** model.

macOS may block the unsigned app. From the app directory:

```bash
xattr -w com.apple.quarantine "0081;65e0bf1c;Terminal;" ./AudioDistressDetector
```

---

## Citation

If you use the models, code, or [deBarbaroFussCry](https://talkbank.org/homebank/access/Password/deBarbaroFussCry.html) data, please cite:

```bibtex
@inproceedings{galhotra26_interspeech,
  title     = {{Advancing Infant Distress Detection: Two- and Three-Way Classification in Real-World Audio Environments}},
  author    = {Yashaswi Galhotra and Priyanka Khante and Anna Madden-Rusnak and Kaya de Barbaro},
  year      = {2026},
  booktitle = {{Interspeech 2026}},
  note      = {Accepted}
}
```

Galhotra, Y., Khante, P., Madden-Rusnak, A., & de Barbaro, K. (2026). Advancing infant distress detection: Two- and three-way classification in real-world audio environments. *Proc. Interspeech 2026* (accepted).

Use of HomeBank data must also follow [TalkBank citation rules](https://talkbank.org/share/irb/options.html) for that corpus.

---

## License and privacy

Models run locally. Fine-tuning and prediction do not upload audio. Do not commit private recordings or label files to this repository.
