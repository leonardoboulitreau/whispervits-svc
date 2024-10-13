import argparse
import pandas as pd
import os, sys
import json
from tqdm import tqdm
from IPython.display import display
from IPython.display import Audio
import torch
import numpy as np
from scipy.io.wavfile import write
import argparse
from pathlib import Path
import librosa
import warnings
warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.getcwd()))
from omegaconf import OmegaConf
from vits.models import SynthesizerInfer
from pitch import load_csv_pitch
from feature_retrieval import IRetrieval, DummyRetrieval, FaissIndexRetrieval, load_retrieve_index
from whisper.model import Whisper, ModelDimensions
from whisper.audio import load_audio, pad_or_trim, log_mel_spectrogram
from hubert import hubert_model
import crepe

def load_whisper_model(path, device) -> Whisper:
    checkpoint = torch.load(path, map_location="cpu")
    dims = ModelDimensions(**checkpoint["dims"])
    # print(dims)
    model = Whisper(dims)
    del model.decoder
    cut = len(model.encoder.blocks) // 4
    cut = -1 * cut
    del model.encoder.blocks[cut:]
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    if not (device == "cpu"):
        model.half()
    model.to(device)
    return model

def pred_ppg(whisper: Whisper, wavPath, ppgPath, device):
    audio = load_audio(wavPath)
    audln = audio.shape[0]
    ppg_a = []
    idx_s = 0
    while (idx_s + 15 * 16000 < audln):
        short = audio[idx_s:idx_s + 15 * 16000]
        idx_s = idx_s + 15 * 16000
        ppgln = 15 * 16000 // 320
        # short = pad_or_trim(short)
        mel = log_mel_spectrogram(short).to(device)
        if not (device == "cpu"):
            mel = mel.half()
        with torch.no_grad():
            mel = mel + torch.randn_like(mel) * 0.1
            ppg = whisper.encoder(mel.unsqueeze(0)).squeeze().data.cpu().float().numpy()
            ppg = ppg[:ppgln,]  # [length, dim=1024]
            ppg_a.extend(ppg)
    if (idx_s < audln):
        short = audio[idx_s:audln]
        ppgln = (audln - idx_s) // 320
        # short = pad_or_trim(short)
        mel = log_mel_spectrogram(short).to(device)
        if not (device == "cpu"):
            mel = mel.half()
        with torch.no_grad():
            mel = mel + torch.randn_like(mel) * 0.1
            ppg = whisper.encoder(mel.unsqueeze(0)).squeeze().data.cpu().float().numpy()
            ppg = ppg[:ppgln,]  # [length, dim=1024]
            ppg_a.extend(ppg)
    np.save(ppgPath, ppg_a, allow_pickle=False)

def load_hubert_audio(file: str, sr: int = 16000):
    x, sr = librosa.load(file, sr=sr)
    return x

def load_hubert_model(path, device):
    model = hubert_model.hubert_soft(path)
    model.eval()
    if not (device == "cpu"):
        model.half()
    model.to(device)
    return model

def pred_vec(model, wavPath, vecPath, device):
    audio = load_hubert_audio(wavPath)
    audln = audio.shape[0]
    vec_a = []
    idx_s = 0
    while (idx_s + 20 * 16000 < audln):
        feats = audio[idx_s:idx_s + 20 * 16000]
        feats = torch.from_numpy(feats).to(device)
        feats = feats[None, None, :]
        if not (device == "cpu"):
            feats = feats.half()
        with torch.no_grad():
            vec = model.units(feats).squeeze().data.cpu().float().numpy()
            vec_a.extend(vec)
        idx_s = idx_s + 20 * 16000
    if (idx_s < audln):
        feats = audio[idx_s:audln]
        feats = torch.from_numpy(feats).to(device)
        feats = feats[None, None, :]
        if not (device == "cpu"):
            feats = feats.half()
        with torch.no_grad():
            vec = model.units(feats).squeeze().data.cpu().float().numpy()
            # print(vec.shape)   # [length, dim=256] hop=320
            vec_a.extend(vec)
    np.save(vecPath, vec_a, allow_pickle=False)


def compute_f0_sing(filename, device):
    audio, sr = librosa.load(filename, sr=16000)
    assert sr == 16000
    audio = torch.tensor(np.copy(audio))[None]
    audio = audio + torch.randn_like(audio) * 0.001
    # Here we'll use a 20 millisecond hop length
    hop_length = 320
    fmin = 50
    fmax = 1000
    model = "full"
    batch_size = 512
    pitch = crepe.predict(
        audio,
        sr,
        hop_length,
        fmin,
        fmax,
        model,
        batch_size=batch_size,
        device=device,
        return_periodicity=False,
    )
    pitch = np.repeat(pitch, 2, -1)  # 320 -> 160 * 2
    pitch = crepe.filter.mean(pitch, 5)
    pitch = pitch.squeeze(0)
    return pitch


def save_csv_pitch(pitch, path):
    with open(path, "w", encoding='utf-8') as pitch_file:
        for i in range(len(pitch)):
            t = i * 10
            minute = t // 60000
            seconds = (t - minute * 60000) // 1000
            millisecond = t % 1000
            print(
                f"{minute}m {seconds}s {millisecond:3d},{int(pitch[i])}", file=pitch_file)


def load_csv_pitch(path):
    pitch = []
    with open(path, "r", encoding='utf-8') as pitch_file:
        for line in pitch_file.readlines():
            pit = line.strip().split(",")[-1]
            pitch.append(int(pit))
    return pitch

def create_retrival(enable_retrieval,spk, retrieval_index_prefix, retrieval_ratio, n_retrieval_vectors) -> IRetrieval:
    if not enable_retrieval:
        return DummyRetrieval()
    else:
        pass
        
    speaker_name = get_speaker_name_from_path(Path(spk))
    base_path = Path(".").absolute() / "data_svc" / "indexes" / speaker_name

    index_name = f"{retrieval_index_prefix}hubert.index"
    hubert_index_filepath = base_path / index_name

    index_name = f"{retrieval_index_prefix}whisper.index"
    whisper_index_filepath = base_path / index_name

    return FaissIndexRetrieval(
        hubert_index=load_retrieve_index(
            filepath=hubert_index_filepath,
            ratio=retrieval_ratio,
            n_nearest_vectors=n_retrieval_vectors
        ),
        whisper_index=load_retrieve_index(
            filepath=whisper_index_filepath,
            ratio=retrieval_ratio,
            n_nearest_vectors=n_retrieval_vectors
        ),
    )


def get_speaker_name_from_path(speaker_path: Path) -> str:
    suffixes = "".join(speaker_path.suffixes)
    filename = speaker_path.name
    return filename.rstrip(suffixes)

def load_svc_model(checkpoint_path, model):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    saved_state_dict = checkpoint_dict["model_g"]
    state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            new_state_dict[k] = saved_state_dict[k]
        except:
            print("%s is not in the checkpoint" % k)
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    return model


def svc_infer(model, retrieval: IRetrieval, spk, pit, ppg, vec, hp, device):
    len_pit = pit.size()[0]
    len_vec = vec.size()[0]
    len_ppg = ppg.size()[0]
    len_min = min(len_pit, len_vec)
    len_min = min(len_min, len_ppg)
    pit = pit[:len_min]
    vec = vec[:len_min, :]
    ppg = ppg[:len_min, :]

    with torch.no_grad():
        spk = spk.unsqueeze(0).to(device)
        source = pit.unsqueeze(0).to(device)
        source = model.pitch2source(source)
        pitwav = model.source2wav(source)
        write("svc_out_pit.wav", hp.data.sampling_rate, pitwav)

        hop_size = hp.data.hop_length
        all_frame = len_min
        hop_frame = 10
        out_chunk = 2500  # 25 S
        out_index = 0
        out_audio = []

        while (out_index < all_frame):

            if (out_index == 0):  # start frame
                cut_s = 0
                cut_s_out = 0
            else:
                cut_s = out_index - hop_frame
                cut_s_out = hop_frame * hop_size

            if (out_index + out_chunk + hop_frame > all_frame):  # end frame
                cut_e = all_frame
                cut_e_out = -1
            else:
                cut_e = out_index + out_chunk + hop_frame
                cut_e_out = -1 * hop_frame * hop_size

            sub_ppg = retrieval.retriv_hubert(ppg[cut_s:cut_e, :])
            sub_vec = retrieval.retriv_whisper(vec[cut_s:cut_e, :])
            sub_ppg = sub_ppg.unsqueeze(0).to(device)
            sub_vec = sub_vec.unsqueeze(0).to(device)
            sub_pit = pit[cut_s:cut_e].unsqueeze(0).to(device)
            sub_len = torch.LongTensor([cut_e - cut_s]).to(device)
            sub_har = source[:, :, cut_s *
                             hop_size:cut_e * hop_size].to(device)
            sub_out = model.inference(
                sub_ppg, sub_vec, sub_pit, spk, sub_len, sub_har)
            sub_out = sub_out[0, 0].data.cpu().detach().numpy()

            sub_out = sub_out[cut_s_out:cut_e_out]
            out_audio.extend(sub_out)
            out_index = out_index + out_chunk

        out_audio = np.asarray(out_audio)
    return out_audio

def convert_TWH(input_wav: str):
    
    # Fixed
    config = "configs/base.yaml"
    model_path = "./lesd5_0100.pth" 
    enable_retrieval = False
    retrieval_index_prefix = ''
    retrieval_ratio =0.5
    n_retrieval_vectors=3

    # SELECT
    input_wav = 'test.wav'
    pitch_shift = -7
    spk_number = 11         # Escolher entre 11 e 20!

    # Set Deivce
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Whisper
    whisper = load_whisper_model(os.path.join("whisper_pretrain", "large-v2.pt"), device)

    # Load Hubert 
    hubert = load_hubert_model(os.path.join("hubert_pretrain", "hubert-soft-0d54a1f4.pt"), device)

    # Load Vits
    hp = OmegaConf.load(config)
    model = SynthesizerInfer(
        hp.data.filter_length // 2 + 1,
        hp.data.segment_size // hp.data.hop_length,
        hp)
    load_svc_model(model_path, model)

    # Create Retrieval
    speaker = "./data_svc_5/singer/" + str(spk_number) + ".spk.npy"
    retrieval = create_retrival(enable_retrieval, speaker, retrieval_index_prefix, retrieval_ratio, n_retrieval_vectors)

    model.eval()
    model.to(device)

    # Set Speaker
    spk = np.load(speaker)
    spk = torch.FloatTensor(spk)

    # Infer Whisper
    ppgPath = "svc_tmp.ppg.npy"
    pred_ppg(whisper, input_wav, ppgPath, device)

    # Load Whisper Out
    ppg = np.load(ppgPath)
    ppg = np.repeat(ppg, 2, 0)  # 320 PPG -> 160 * 2
    ppg = torch.FloatTensor(ppg)

    # Infer Hubert
    vecPath = "svc_tmp.vec.npy"
    pred_vec(hubert, input_wav, vecPath, device)

    # Load Hubert Out
    vec = np.load(vecPath)
    vec = np.repeat(vec, 2, 0)  # 320 PPG -> 160 * 2
    vec = torch.FloatTensor(vec)

    # Set and Shift Pitch
    pitPath = "svc_tmp.pit.csv"
    pitch = compute_f0_sing(input_wav, device)
    save_csv_pitch(pitch, pitPath)
    pit = load_csv_pitch(pitPath)
    if (pitch_shift == 0):
        pass
    else:
        pit = np.array(pit)
        source = pit[pit > 0]
        source_ave = source.mean()
        source_min = source.min()
        source_max = source.max()
        shift = pitch_shift
        shift = 2 ** (shift / 12)
        pit = pit * shift
    pit = torch.FloatTensor(pit)

    out_audio = svc_infer(model, retrieval, spk, pit, ppg, vec, hp, device)
    print('ok')
    return out_audio

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("twh_path", type=str)
    args = parser.parse_args()
    
    convert_TWH(args.twh_path)

if __name__ == "__main__":
    main()
