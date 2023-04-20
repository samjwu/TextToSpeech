"""
Visualizes text.
Generates spectrograms and waveforms from text.
"""

import matplotlib
import matplotlib.pyplot
import torch
import torchaudio

def text_to_sequence(text: str, symbol_table: dict[str, int]):
    """
    Map characters in a given string to characters in a table.
    """
    text = text.lower()
    return [symbol_table[s] for s in text if s in symbols]

def character_encoding(processor_wavernn_char, sample_text: str):
    """
    Convert a given string to an encoded string using character-based WaveRNN text processor.
    """
    processed_wavernn_char, lengths = processor_wavernn_char(sample_text)
    print(f"Character Encoding: {processed_wavernn_char}")
    tokens = [processor_wavernn_char.tokens[i] for i in processed_wavernn_char[0, : lengths[0]]]
    print(f"Tokens: {tokens}\n")
    return (processed_wavernn_char, lengths)

def phoneme_encoding(processor_wavernn_phon, sample_text: str):
    """
    Convert a given string to an encoded string using phoneme-based WaveRNN text processor.
    """
    with torch.inference_mode():
        processed_wavernn_phon, lengths = processor_wavernn_phon(sample_text)
    print(f"Phoneme Encoding: {processed_wavernn_phon}")
    tokens = [processor_wavernn_phon.tokens[i] for i in processed_wavernn_phon[0, : lengths[0]]]
    print(f"Tokens: {tokens}\n")
    return (processed_wavernn_phon, lengths)

def generate_spectrogram(processed_wavernn_phon, lengths, tacotron2):
    """
    Generate spectrogram from encoded text.
    Visual representation of frequency against time.
    """
    spectrogram, _, _ = tacotron2.infer(processed_wavernn_phon, lengths)
    matplotlib.pyplot.imshow(spectrogram[0].cpu().detach(), origin="lower", aspect="auto")
    matplotlib.pyplot.show()

def generate_spectrograms(processed_wavernn_phon, lengths, tacotron2, number):
    """
    Generate a given number of spectrograms from encoded text.
    Due to use of multinomial sampling, generated spectrograms are random and have slight variations.
    """
    _, axes = matplotlib.pyplot.subplots(3, 1, figsize=(16, 4.3 * 3))
    for i in range(number):
        with torch.inference_mode():
            spectrograms, _, _ = tacotron2.infer(processed_wavernn_phon, lengths)
        print(f"Spectrogram {i} shape: {spectrograms[0].shape}")
        axes[i].imshow(spectrograms[0].cpu().detach(), origin="lower", aspect="auto")
    matplotlib.pyplot.show()

def generate_waveform(device, sample_text: str, processor_wavernn_phon, tacotron2, vocoder):
    """
    Encode a given string and then generate a spectrogram from the encoded text.
    Then convert the spectrogram into a waveform.
    Visual representation of amplitude or shape against time.
    """
    with torch.inference_mode():
        processed_wavernn_phon, lengths = processor_wavernn_phon(sample_text)
        processed_wavernn_phon = processed_wavernn_phon.to(device)
        lengths = lengths.to(device)
        spectrogram, spectrogram_lengths, _ = tacotron2.infer(processed_wavernn_phon, lengths)
        waveforms, _ = vocoder(spectrogram, spectrogram_lengths)
    _, [axes1, axes2] = matplotlib.pyplot.subplots(2, 1, figsize=(16, 9))
    axes1.imshow(spectrogram[0].cpu().detach(), origin="lower", aspect="auto")
    axes2.plot(waveforms[0].cpu().detach())
    matplotlib.pyplot.show()

if __name__ == "__main__":
    # init configs
    matplotlib.rcParams["figure.figsize"] = [16.0, 4.8]
    torch.random.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # check versions and device used
    print(f"torch version: {torch.__version__}")
    print(f"torchaudio version: {torchaudio.__version__}")
    print(f"device: {device}\n")

    # prepare symbols used for encoding
    symbols = "_-!'(),.:;? abcdefghijklmnopqrstuvwxyz"
    symbol_table = {s: i for i, s in enumerate(symbols)}
    symbols = set(symbols)

    sample_text = "Hello World"
    print(f"Original Text: {sample_text}")
    print(f"Encoded Text: {text_to_sequence(sample_text, symbol_table)}\n")

    # Pretrained models: https://pytorch.org/audio/main/pipelines.html#id60

    # character encoding
    bundle_wavernn_char = torchaudio.pipelines.TACOTRON2_WAVERNN_CHAR_LJSPEECH
    processor_wavernn_char = bundle_wavernn_char.get_text_processor()
    character_encoding(processor_wavernn_char, sample_text)

    # phoneme encoding
    bundle_wavernn_phon = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH
    processor_wavernn_phon = bundle_wavernn_phon.get_text_processor()
    (processed_wavernn_phon, lengths) = phoneme_encoding(processor_wavernn_phon, sample_text)

    # spectrograms
    tacotron2_wavernn = bundle_wavernn_phon.get_tacotron2().to(device)
    processed_wavernn_phon = processed_wavernn_phon.to(device)
    lengths = lengths.to(device)
    generate_spectrogram(processed_wavernn_phon, lengths, tacotron2_wavernn)
    generate_spectrograms(processed_wavernn_phon, lengths, tacotron2_wavernn, 3)

    # waveforms
    vocoder_wavernn = bundle_wavernn_phon.get_vocoder().to(device)
    generate_waveform(device, sample_text, processor_wavernn_phon, tacotron2_wavernn, vocoder_wavernn)
    
    bundle_griffinlim = torchaudio.pipelines.TACOTRON2_GRIFFINLIM_PHONE_LJSPEECH
    processor_griffinlim = bundle_griffinlim.get_text_processor()
    tacotron2_griffinlim = bundle_griffinlim.get_tacotron2().to(device)
    vocoder_griffinlim = bundle_griffinlim.get_vocoder().to(device)
    generate_waveform(device, sample_text, processor_griffinlim, tacotron2_griffinlim, vocoder_griffinlim)
    