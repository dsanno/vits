import argparse
import json

import soundfile as sf
import torch

import mel_processing
import utils
from models import VoiceConverter

torch.backends.cudnn.benchmark = True


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('config_path', help='Configuration file path')
  parser.add_argument('model_path', help='Generator model file path')
  parser.add_argument('input_audio_path', help='Input audio file path')
  parser.add_argument('output_audio_path', help='Output audio file path')
  parser.add_argument('source_speaker_id', type=int, help='Source speaker ID')
  parser.add_argument('target_speaker_id', type=int, help='Target speaker ID')
  return parser.parse_args()


def main():
  args = parse_args()

  device = torch.device('cuda')
  hps = utils.get_hparams_from_file(args.config_path)
  net_g = VoiceConverter(
      hps.data.filter_length // 2 + 1,
      hps.train.segment_size // hps.data.hop_length,
      n_speakers=hps.data.n_speakers,
      **hps.model,
  )
  utils.load_checkpoint(args.model_path, net_g)
  net_g = net_g.to(device)
  net_g.eval()

  audio, sampling_rate = utils.load_wav_to_torch(args.input_audio_path)
  audio_norm = audio / hps.data.max_wav_value
  audio_norm = audio_norm.unsqueeze(0)
  spec = mel_processing.spectrogram_torch(audio_norm, hps.data.filter_length, hps.data.sampling_rate,
                                          hps.data.hop_length, hps.data.win_length, center=False)
  spec = spec.to(device)
  spec_length = torch.tensor([spec.size(2)], dtype=torch.int64, device=device)
  source_speaker_id = torch.tensor([args.source_speaker_id], dtype=torch.int64, device=device)
  target_speaker_id = torch.tensor([args.target_speaker_id], dtype=torch.int64, device=device)
  with torch.no_grad():
    output_wave, _, _ = net_g(spec, spec_length, source_speaker_id, target_speaker_id)
  sf.write(args.output_audio_path, output_wave.cpu().numpy()[0, 0], sampling_rate)


if __name__ == "__main__":
  main()
