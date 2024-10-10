from faster_whisper import WhisperModel
from datasets import load_dataset

model = WhisperModel("arc-r/faster-whisper-large-v2-Ko")
ds = load_dataset("ricecake/Genshin_Impact_RaidenShogun_Voice_korean")
ds2 = load_dataset("Bingsu/KSS_Dataset")

# print(ds["train"]["audio"][0])
# print(ds2["train"]["audio"][0])
# print(ds2["train"]["original_script"][0])
# print(ds2["train"]["duration"][0])

segments, info = model.transcribe("datasets/moon/audio/0_dahye_1.wav")


for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
