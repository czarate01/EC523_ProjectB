# Project B command-line interface

This starter enforces one simple contract: one checkpoint maps each mono
16 kHz input waveform to an equal-length waveform. Students replace only
`submission/model.py` and provide `submission/model.safetensors`. Staff own
the runner and evaluator for official submissions.

## Demo

```sh
python make_demo.py --output demo
python enhance.py --checkpoint demo/model.safetensors \
  --input demo/noisy --output demo/enhanced --device cpu
python -m unittest discover -s tests -v
```

The generated signals are synthetic tones, not speech. The included spectral
gate is untrained and close to an identity transform. Neither is a baseline
result.

## Official inference

```sh
python enhance.py --checkpoint submission/model.safetensors \
  --input evaluation/noisy --output predictions --device cuda
```

`enhance.py` imports `build_enhancer(16000)`, loads safe tensors, and invokes
`model.enhance(waveform, 16000)` for each file. The model receives a `[1,T]`
float32 tensor only. It receives no filename, clean target, speaker, SNR,
transcript, or score. Outputs must be finite mono 16 kHz WAVs with the same
sample count and peak magnitude no greater than 1.0. The runner never clips,
normalizes, resamples, pads, trims, or aligns student output.

## Evaluation

```sh
python evaluate.py --clean evaluation/clean \
  --enhanced predictions --report noisy_report.json
python score.py --noisy-report noisy_report.json \
  --clean-report clean_preservation_report.json
```

The public aggregation and SI-SDR implementation are included. The environment
pins the public packages [`pesq==0.0.4`](https://pypi.org/project/pesq/0.0.4/)
and `pystoi==0.4.1`. Official runs use those exact versions; changing a backend
can make scores incomparable.

The exact score is `100 * (0.85 * noisy_quality + 0.15 * clean_quality)`, where
each quality is `0.40 * normalized_PESQ + 0.30 * normalized_STOI + 0.30 *
normalized_SI_SDR`. Read Sections 6--8 of the handout before interpreting it.
