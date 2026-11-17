# Licence of `hey_cortex.onnx`

**This one file is not covered by the repository's MIT licence.**

`hey_cortex.onnx` (the custom "Hey Cortex" wake-word model) is licensed under
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
(CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).
You may use and share it for non-commercial purposes, with attribution, and
derivatives must keep the same licence.

## Why

The model was trained with the
[`briankelley/atlas-voice-training`](https://github.com/briankelley/atlas-voice-training)
pipeline (Apache-2.0), a packaging of the
[openWakeWord](https://github.com/dscripka/openWakeWord) training process. Its
negative training data are openWakeWord's precomputed ACAV100M features
([`davidscripka/openwakeword_features`](https://huggingface.co/datasets/davidscripka/openwakeword_features)),
released under CC BY-NC-SA 4.0, and the pipeline states that models trained
with it inherit that non-commercial restriction. openWakeWord's own pre-trained
models carry the same licence for the same reason.

## Attribution

- Training pipeline: atlas-voice-training by Brian Kelley (Apache-2.0).
- Negative features: openWakeWord ACAV100M features by David Scripka
  (CC BY-NC-SA 4.0).
- Background audio for augmentation: MUSAN corpus (CC BY 4.0).
- Room impulse responses: MIT Room Impulse Response survey.
- Positive samples: synthetic speech generated with Piper TTS via the
  piper-sample-generator.

Change made: a new wake-word model trained for the phrase "hey cortex".

Everything else in this repository remains under the MIT licence in
[`LICENSE`](../../../LICENSE).
