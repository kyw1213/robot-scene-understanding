# Robot Scene Understanding

  Indoor scene understanding practice for robot navigation.

  ## Goals

  - Train ResNet18/ResNet50 on MIT Indoor 67.
  - Run CLIP zero-shot scene classification.
  - Compare supervised CNN and VLM-based zero-shot recognition.
  - Analyze failure cases for robot navigation scenarios.

  ## Structure

  ```text
  configs/
  src/
  outputs/
  notes/

  ## Dataset

  MIT Indoor 67 should be placed outside this repository, for example:

  ~/datasets/mit_indoor67/
