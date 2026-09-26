"""
Run per-second predictions with a released or fine-tuned model.

  python predict.py --audio examples/test.wav --mode binary
  python predict.py --audio examples/test.wav --mode ternary
"""

import argparse
import os
from collections import Counter

import joblib
import librosa
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

SAMPLE_RATE = 16000
CHUNK_SIZE = 5 * SAMPLE_RATE
OVERLAP = 4 * SAMPLE_RATE
BINARY_NAMES = {0: "non-distress", 1: "distress"}
TERNARY_NAMES = {0: "non-distress", 1: "fuss", 2: "cry"}


def generate_chunks(audio):
    chunks, start = [], 0
    while start + CHUNK_SIZE <= len(audio):
        chunks.append(audio[start : start + CHUNK_SIZE])
        start += CHUNK_SIZE - OVERLAP
    if start < len(audio) and len(audio) - start >= SAMPLE_RATE:
        chunks.append(audio[start:])
    return chunks


def extract_features(chunks, yamnet_model):
    embeddings = []
    for chunk in chunks:
        waveform = tf.convert_to_tensor(chunk, dtype=tf.float32)
        pad = max(0, SAMPLE_RATE - tf.shape(waveform)[0])
        waveform = tf.concat([waveform, tf.zeros([pad], tf.float32)], 0)[:SAMPLE_RATE]
        _, emb, _ = yamnet_model(tf.reshape(waveform, [-1]))
        embeddings.append(emb.numpy()[0])
    return np.array(embeddings)


def drop_short_distress(labels, min_sec=3):
    """Set fuss/cry runs shorter than min_sec to 0. Optional post-filter, not used in the paper."""
    out = [int(x) for x in labels]
    i, n = 0, len(out)
    while i < n:
        if out[i] == 0:
            i += 1
            continue
        j = i
        while j < n and out[j] != 0:
            j += 1
        if j - i < min_sec:
            out[i:j] = [0] * (j - i)
        i = j
    return out


def majority_vote(preds, chunk_size=5, overlap=4):
    per = {}
    hop = chunk_size - overlap
    for i, pred in enumerate(preds):
        for second in range(i * hop, i * hop + chunk_size):
            per.setdefault(second, []).append(int(pred))
    voted = []
    for second in range(max(per) + 1):
        if second in per:
            voted.append(Counter(per[second]).most_common(1)[0][0])
        else:
            voted.append(voted[-1] if voted else 0)
    return np.array(voted)


def main():
    parser = argparse.ArgumentParser(description="Predict infant distress labels")
    parser.add_argument("--audio", required=True, help="Path to a WAV file")
    parser.add_argument(
        "--model_dir",
        default=None,
        help="Folder with svm_model.pkl, scaler.pkl, pca.pkl "
        "(default: weights/binary or weights/ternary from --mode)",
    )
    parser.add_argument(
        "--mode",
        choices=["binary", "ternary"],
        default="binary",
        help="binary: 0/1 distress. ternary: 0/1/2 fuss vs cry",
    )
    parser.add_argument(
        "--min_distress_sec",
        type=int,
        default=0,
        help="If > 0, drop fuss/cry runs shorter than this many seconds "
        "(optional cleanup; 0 keeps the paper output)",
    )
    args = parser.parse_args()
    if args.model_dir is None:
        args.model_dir = os.path.join("weights", args.mode)

    names = TERNARY_NAMES if args.mode == "ternary" else BINARY_NAMES
    svm = joblib.load(os.path.join(args.model_dir, "svm_model.pkl"))
    scaler = joblib.load(os.path.join(args.model_dir, "scaler.pkl"))
    pca = joblib.load(os.path.join(args.model_dir, "pca.pkl"))
    yamnet = hub.KerasLayer("https://tfhub.dev/google/yamnet/1")

    audio, _ = librosa.load(args.audio, sr=SAMPLE_RATE)
    chunks = generate_chunks(audio)
    if not chunks:
        raise SystemExit("Audio is too short (need at least 1 second).")

    raw = svm.predict(pca.transform(scaler.transform(extract_features(chunks, yamnet))))
    voted = majority_vote(raw)
    n = min(len(voted), int(len(audio) / SAMPLE_RATE) or len(voted))
    voted = voted[:n]
    if args.min_distress_sec > 0:
        voted = drop_short_distress(voted, args.min_distress_sec)

    print(f"File: {args.audio}")
    print(f"Seconds: {n}  |  SVM classes: {list(svm.classes_)}")
    counts = Counter(voted.tolist())
    for cls in sorted(counts):
        print(f"  {names.get(cls, cls)}: {counts[cls]}s")

    if args.mode == "ternary":
        distress = int(np.sum(voted != 0))
        print(f"  distress (fuss+cry): {distress}s")

    print("\nPer-second labels:")
    for second, pred in enumerate(voted):
        print(f"  {second:5d}  {pred}  {names.get(int(pred), pred)}")


if __name__ == "__main__":
    main()
