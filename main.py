"""
Text to Speech
"""

import matplotlib
import torch
import torchaudio

def text_to_sequence(text: str, symbol_table: dict[str, int]):
    text = text.lower()
    return [symbol_table[s] for s in text if s in symbols]

matplotlib.rcParams["figure.figsize"] = [16.0, 4.8]

torch.random.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"torch version: {torch.__version__}")
print(f"torchaudio version: {torchaudio.__version__}")
print(f"device: {device}\n")

symbols = "_-!'(),.:;? abcdefghijklmnopqrstuvwxyz"
symbol_table = {s: i for i, s in enumerate(symbols)}
symbols = set(symbols)

sample_text = "TextToSpeech example: Hello World"
print(f"Original Text: {sample_text}")
print(f"Encoded Text: {text_to_sequence(sample_text, symbol_table)}\n")

# Pretrained models: https://pytorch.org/audio/main/pipelines.html#id60

# character encoding
processor_wavernn_char = torchaudio.pipelines.TACOTRON2_WAVERNN_CHAR_LJSPEECH.get_text_processor()
processed_wavernn_char, lengths = processor_wavernn_char(sample_text)
print(f"Character Encoding: {processed_wavernn_char}")
tokens = [processor_wavernn_char.tokens[i] for i in processed_wavernn_char[0, : lengths[0]]]
print(f"Tokens: {tokens}\n")

# phoneme encoding
processor_wavernn_phon = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH.get_text_processor()
with torch.inference_mode():
    processed_wavernn_phon, _ = processor_wavernn_phon(sample_text)
print(f"Phoneme Encoding: {processed_wavernn_phon}")
