"""
Fine-tune the released 2-way (binary) infant distress model on new data.

Reuses the pretrained scaler and PCA, then trains a new RBF SVM.
Classes: 0 = non-distress, 1 = distress (fuss + cry + scream).

CSV format (no header): start,end,label
  label = 0/1/2 or fuss/cry  (1 and 2 both become distress)

Example:
  python fine_tune_binary.py \\
    --model_path weights/binary \\
    --data_folder /path/to/audio \\
    --label_folder /path/to/labels \\
    --output_path ./finetuned_binary
"""

import argparse
import json
import os
import warnings
from collections import Counter
from datetime import datetime

import joblib
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from imblearn.under_sampling import RandomUnderSampler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

SAMPLE_RATE = 16000
CHUNK_SIZE = 5 * SAMPLE_RATE
OVERLAP = 4 * SAMPLE_RATE


def label_to_num(label):
    """Binary: 0 non-distress, 1 distress (fuss or cry)."""
    if isinstance(label, (int, float)) and not isinstance(label, bool):
        return 1 if int(label) in (1, 2) else 0

    text = str(label).strip().lower()
    try:
        return 1 if int(float(text)) in (1, 2) else 0
    except ValueError:
        pass

    if text in ("fuss", "cry", "scream", "distress"):
        return 1
    return 0


def index_csvs(label_folder):
    index = {}
    for dirpath, _, filenames in os.walk(label_folder):
        for name in filenames:
            if name.endswith(".csv") and not name.startswith("._"):
                index[name[:-4]] = os.path.join(dirpath, name)
    return index


def collect_pairs(data_folder, label_folder):
    csv_index = index_csvs(label_folder)
    pairs, missing = [], []
    for dirpath, _, filenames in os.walk(data_folder):
        for name in sorted(filenames):
            if not name.endswith(".wav") or name.startswith("._"):
                continue
            stem = name[:-4]
            audio_path = os.path.join(dirpath, name)
            label_path = csv_index.get(stem)
            if label_path:
                pairs.append((audio_path, label_path, stem))
            else:
                missing.append(os.path.relpath(audio_path, data_folder))
    return pairs, missing


class FineTuneInfantDistressDetector:
    def __init__(self, model_path):
        self.model_path = model_path
        self.yamnet_model = hub.KerasLayer("https://tfhub.dev/google/yamnet/1")
        print("YAMNet model loaded successfully.")
        self.load_pretrained_model()

    def load_pretrained_model(self):
        self.pretrained_svm = joblib.load(os.path.join(self.model_path, "svm_model.pkl"))
        self.pretrained_scaler = joblib.load(os.path.join(self.model_path, "scaler.pkl"))
        self.pretrained_pca = joblib.load(os.path.join(self.model_path, "pca.pkl"))
        print("Loaded pretrained svm_model.pkl, scaler.pkl, pca.pkl")
        print(f"  Released SVM classes: {list(self.pretrained_svm.classes_)}")

    def generate_chunks(self, audio, start, end, sr=SAMPLE_RATE):
        chunks = []
        current_start = int(start * sr)
        stop = min(int(end * sr), len(audio))
        while current_start + CHUNK_SIZE <= stop:
            chunks.append(audio[current_start : current_start + CHUNK_SIZE])
            current_start += CHUNK_SIZE - OVERLAP
        if current_start < stop:
            last_chunk = audio[current_start:stop]
            if len(last_chunk) >= SAMPLE_RATE:
                chunks.append(last_chunk)
        return chunks

    def extract_features(self, chunks, sample_rate=SAMPLE_RATE):
        all_embeddings = []
        for chunk in chunks:
            waveform = tf.convert_to_tensor(chunk, dtype=tf.float32)
            padding_needed = max(0, sample_rate - tf.shape(waveform)[0])
            zero_padding = tf.zeros([padding_needed], dtype=tf.float32)
            waveform = tf.concat([waveform, zero_padding], 0)
            waveform = waveform[:sample_rate]
            waveform = tf.reshape(waveform, [-1])
            scores, embeddings, spectrogram = self.yamnet_model(waveform)
            all_embeddings.append(embeddings.numpy()[0])
        return all_embeddings

    def process_new_dataset(self, data_folder, label_folder):
        pairs, missing = collect_pairs(data_folder, label_folder)
        print(f"Matched wav+csv pairs: {len(pairs)}")
        if missing:
            print(f"WAV files with no CSV (skipped): {len(missing)}")

        embeddings, labels = [], []
        for audio_path, label_path, stem in pairs:
            print(f"  Processing: {stem}")
            audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
            df_events = pd.read_csv(
                label_path, header=None, names=["start", "end", "label"]
            )
            df_events.sort_values(by="start", inplace=True)
            previous_end = 0.0

            for _, row in df_events.iterrows():
                start, end = float(row["start"]), float(row["end"])
                if end <= start:
                    continue
                if start - previous_end > 1:
                    gap_chunks = self.generate_chunks(audio, previous_end, start, sr)
                    if gap_chunks:
                        embeddings.extend(self.extract_features(gap_chunks))
                        labels.extend([0] * len(gap_chunks))
                event_chunks = self.generate_chunks(audio, start, end, sr)
                if event_chunks:
                    embeddings.extend(self.extract_features(event_chunks))
                    labels.extend([label_to_num(row["label"])] * len(event_chunks))
                previous_end = end

            audio_end = len(audio) / sr
            if audio_end - previous_end > 1:
                gap_chunks = self.generate_chunks(audio, previous_end, audio_end, sr)
                if gap_chunks:
                    embeddings.extend(self.extract_features(gap_chunks))
                    labels.extend([0] * len(gap_chunks))

        print(f"Processed {len(embeddings)} samples")
        print(f"Label distribution: {Counter(labels)}")
        return np.array(embeddings), np.array(labels)

    def fine_tune_model(self, new_embeddings, new_labels):
        scaled = self.pretrained_scaler.transform(new_embeddings)
        reduced = self.pretrained_pca.transform(scaled)
        print("Balancing dataset (undersampling)...")
        sampler = RandomUnderSampler(random_state=42)
        reduced, new_labels = sampler.fit_resample(reduced, new_labels)
        print(f"Balanced dataset: {Counter(new_labels)}")
        print("Training fine-tuned RBF SVM...")
        model = SVC(kernel="rbf", probability=True, random_state=42)
        model.fit(reduced, new_labels)
        print(f"SVM classes_: {list(model.classes_)}")
        return model

    def save_finetuned_model(self, model, output_path):
        os.makedirs(output_path, exist_ok=True)
        joblib.dump(model, os.path.join(output_path, "svm_model.pkl"))
        joblib.dump(self.pretrained_scaler, os.path.join(output_path, "scaler.pkl"))
        joblib.dump(self.pretrained_pca, os.path.join(output_path, "pca.pkl"))
        metadata = {
            "task": "binary_infant_distress",
            "classes": {"0": "non-distress", "1": "distress"},
            "original_model_path": os.path.abspath(self.model_path),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        with open(os.path.join(output_path, "finetuning_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved fine-tuned 2-way model to: {output_path}")
        print("  svm_model.pkl, scaler.pkl, pca.pkl")
        return metadata["created_at"]

    def run_finetuning(self, data_folder, label_folder, output_path):
        embeddings, labels = self.process_new_dataset(data_folder, label_folder)
        if len(embeddings) == 0:
            print("Error: no training samples found.")
            return None
        model = self.fine_tune_model(embeddings, labels)
        return self.save_finetuned_model(model, output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune the 2-way (distress vs non-distress) model"
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help="Folder with released svm_model.pkl, scaler.pkl, pca.pkl",
    )
    parser.add_argument("--data_folder", required=True, help="Audio root (participant folders)")
    parser.add_argument("--label_folder", required=True, help="Label CSV root")
    parser.add_argument("--output_path", default="./finetuned_binary")
    args = parser.parse_args()
    FineTuneInfantDistressDetector(args.model_path).run_finetuning(
        args.data_folder, args.label_folder, args.output_path
    )
    print("Binary fine-tuning completed.")


if __name__ == "__main__":
    main()
