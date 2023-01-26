""" Warmup whisper, CTC-HuBERT aligner, and silero VAD so that docker caches checkpoints. """

import time

import nltk
import torch
import whisperx
from transformers import HubertForCTC, Wav2Vec2Processor

print('[WARMUP] torch.cuda.is_available: ', torch.cuda.is_available())
time.sleep(1)


# loads and caches the whisper at Docker start to prevent long first load on server-side
model = whisperx.load_model('tiny')
print("Warmup: Whisper loaded. Now performing practice inference.")
print(f"Warmup: model is on device {next(iter(model.parameters())).device}")

result = model.transcribe('test.wav', without_timestamps=False)

print("Whisper warmed up! Now warming up HuBERT.")
# loads and cache HuBERT for CTC
# _ = HubertForCTC.from_pretrained(f"facebook/hubert-xlarge-ls960-ft")
# _ = Wav2Vec2Processor.from_pretrained(f"facebook/hubert-xlarge-ls960-ft")
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device='cpu')


# Warmup silero
_, _ = torch.hub.load(
    repo_or_dir="snakers4/silero-vad", model="silero_vad", onnx=False
)

# # download punkt for nltk to work.
nltk.download('punkt')
