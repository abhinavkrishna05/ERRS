!pip install -q tensorflow-hub faster-whisper ctranslate2 sentencepiece accelerate librosa soundfile ipywidgets

import shutil, os
src = "/kaggle/input/datasets/meghajha18/sample-audio/audio/clean"
os.makedirs("audio/clean", exist_ok=True)
shutil.copytree(src, "audio/clean", dirs_exist_ok=True)
print("Clean files copied:", len(os.listdir("audio/clean")))

import os, io, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np, librosa, matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, Audio, clear_output

# ---- Loading all 3 models ONCE ----
if "yamnet_model" not in globals():
    import tensorflow as tf, tensorflow_hub as hub, csv
    print("Loading YAMNet...")
    yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
    class_map_path = yamnet_model.class_map_path().numpy().decode("utf-8")
    with tf.io.gfile.GFile(class_map_path) as f:
        yamnet_classes = [row["display_name"] for row in csv.DictReader(f)]

if "whisper_model" not in globals():
    from faster_whisper import WhisperModel
    print("Loading faster-whisper (medium)...")
    try:
        whisper_model = WhisperModel("medium", device="cuda", compute_type="float16")
    except Exception:
        whisper_model = WhisperModel("medium", device="cpu", compute_type="int8")

if "nllb_model" not in globals():
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    print("Loading NLLB-200...")
    NLLB_NAME = "facebook/nllb-200-distilled-600M"
    nllb_tokenizer = AutoTokenizer.from_pretrained(NLLB_NAME)
    nllb_model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_NAME)
    nllb_device = "cuda" if torch.cuda.is_available() else "cpu"
    nllb_model = nllb_model.to(nllb_device)
    print("All models ready.\n")

WHISPER_TO_FLORES = {"en": "eng_Latn", "hi": "hin_Deva", "ur": "urd_Arab",
                      "bn": "ben_Beng", "ta": "tam_Taml", "mr": "mar_Deva"}

def translate_to_english(text, whisper_lang):
    if not text.strip() or whisper_lang == "en":
        return text, False
    flores = WHISPER_TO_FLORES.get(whisper_lang, "hin_Deva")
    nllb_tokenizer.src_lang = flores
    inputs = nllb_tokenizer(text, return_tensors="pt").to(nllb_device)
    forced_id = nllb_tokenizer.convert_tokens_to_ids("eng_Latn")
    out = nllb_model.generate(**inputs, forced_bos_token_id=forced_id, max_length=256)
    return nllb_tokenizer.batch_decode(out, skip_special_tokens=True)[0], True

def run_demo(audio_arr: np.ndarray, title: str):
    with output_area:
        clear_output(wait=True)
        print(f"=== {title} ===\n")

        # Waveform + player
        plt.figure(figsize=(10, 2.5))
        plt.plot(audio_arr, linewidth=0.5, color="#065A82")
        plt.title("Waveform"); plt.tight_layout(); plt.show()
        display(Audio(audio_arr, rate=16000))

        # --- YAMNet:for acoustics ---
        scores, embeddings, mel = yamnet_model(audio_arr.astype(np.float32))
        mean_scores = tf.reduce_mean(scores, axis=0).numpy()
        top_idx = np.argsort(mean_scores)[::-1][:5]
        print("YAMNet top acoustic tags (works even with background noise):")
        for i in top_idx:
            print(f"   {yamnet_classes[i]:25s} {mean_scores[i]:.3f}")

        # --- Whisper:for transcription ---
        segments, info = whisper_model.transcribe(audio_arr, beam_size=5, vad_filter=True)
        transcript = " ".join(s.text.strip() for s in segments)
        print(f"\nDetected language: {info.language} (p={info.language_probability:.2f})")
        print(f"Transcript: {transcript}")

        # --- NLLB: for translation to English ---
        translated, was_translated = translate_to_english(transcript, info.language)
        print(f"\nEnglish translation{' (translated)' if was_translated else ' (already English)'}:")
        print(f"   {translated}")

def _get_uploaded_bytes(w):
    val = w.value
    if not val: return None
    item = list(val.values())[0] if isinstance(val, dict) else val[0]
    return item["content"]

def on_upload_change(change):
    content = _get_uploaded_bytes(upload_widget)
    if content is None: return
    audio_arr, _ = librosa.load(io.BytesIO(bytes(content)), sr=16000, mono=True)
    run_demo(audio_arr, "Your uploaded audio")

def on_sample_run(_):
    path = sample_dropdown.value
    audio_arr, _ = librosa.load(path, sr=16000, mono=True)
    run_demo(audio_arr, f"Sample: {os.path.basename(path)}")

# ----  sample dropdown from already-generated noisy audio ----
candidates = []
for pattern in ["audio/augmented/*_noise.wav", "audio/augmented/*_telephone.wav",
                "audio/augmented/*_background.wav", "audio/augmented/*_reverb.wav"]:
    candidates += sorted(glob.glob(pattern))[:2]
if not candidates:
    candidates = sorted(glob.glob("audio/clean/*.wav"))[:5]

sample_dropdown = widgets.Dropdown(options=candidates, description="Sample:")
run_button = widgets.Button(description="Run on selected sample")
run_button.on_click(on_sample_run)

upload_widget = widgets.FileUpload(accept="audio/*", multiple=False, description="Upload audio")
upload_widget.observe(on_upload_change, names="value")

output_area = widgets.Output()

print("Pick a pre-made noisy sample OR upload your own audio file, then watch YAMNet -> Whisper -> NLLB run:")
display(widgets.HBox([sample_dropdown, run_button]), upload_widget, output_area)
