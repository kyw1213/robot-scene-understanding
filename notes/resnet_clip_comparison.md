  # MIT Indoor 67: ResNet and CLIP Comparison Note

  ## 1. Experiment Goal

  This experiment evaluates indoor scene recognition methods on the MIT Indoor 67 dataset.

  The goal is to compare traditional supervised CNN fine-tuning with vision-language zero-shot recognition, and understand their differences for robot scene understanding and semantic navigation.

  ## 2. Dataset

  Dataset: MIT Indoor 67

  The dataset contains 67 indoor scene categories, such as bedroom, kitchen, corridor, office, library, restaurant, and living room.

  Official split:

  - Train set: MIT Indoor 67 official train split
  - Test set: MIT Indoor 67 official test split

  ## 3. Methods

  ### 3.1 ResNet18 Fine-tuning

  ResNet18 is initialized with ImageNet pretrained weights and fine-tuned on the MIT Indoor 67 training set.

  Training settings:

  - Input size: 224 x 224
  - Optimizer: AdamW
  - Learning rate: 1e-4
  - Weight decay: 1e-4
  - Loss: Cross entropy with label smoothing
  - Training data: MIT Indoor 67 train split

  ### 3.2 ResNet50 Fine-tuning

  ResNet50 is also initialized with ImageNet pretrained weights and fine-tuned on the same training split.

  Compared with ResNet18, ResNet50 has stronger representation capacity, but also requires more computation and is more prone to overfitting on small datasets if not regularized properly.

  ### 3.3 CLIP Zero-shot

  CLIP ViT-B/32 is used as a zero-shot baseline.

  For each class, a text prompt is constructed:

  ```text
  a photo of a {class_name}

  The image embedding and text embeddings are compared by cosine similarity. The class with the highest similarity is used as the prediction.

  Unlike ResNet18 and ResNet50, CLIP is not trained on the MIT Indoor 67 training set in this experiment.

  ## 4. Results

   Method                  Backbone    Training on MIT Indoor 67    Accuracy
  ━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━
   ResNet18 fine-tuning    ResNet18                          Yes      72.91%
  ──────────────────────  ──────────  ───────────────────────────  ──────────
   ResNet50 fine-tuning    ResNet50                          Yes      77.91%
  ──────────────────────  ──────────  ───────────────────────────  ──────────
   CLIP zero-shot          ViT-B/32                           No      81.19%

  ## 5. Main Observations

  ### 5.1 ResNet50 improves over ResNet18

  ResNet50 achieves 77.91% accuracy, higher than ResNet18's 75.22%.

  This suggests that a stronger CNN backbone improves indoor scene recognition performance. The improvement is reasonable because ResNet50 has deeper layers and stronger feature representation than ResNet18.

  However, the improvement is moderate, not dramatic. This may be because MIT Indoor 67 is relatively small, and simply increasing CNN capacity cannot fully solve the semantic ambiguity between indoor scenes.

  ### 5.2 CLIP zero-shot outperforms both supervised CNN baselines

  CLIP zero-shot achieves 81.19% accuracy, outperforming both ResNet18 and ResNet50, even though it is not fine-tuned on MIT Indoor 67.

  This indicates that large-scale vision-language pretraining provides strong semantic representations for indoor scene recognition.

  Indoor scene categories are naturally language-level concepts, such as kitchen, bedroom, office, and library. CLIP has likely learned these visual-language associations from large-scale image-text pretraining, which helps it generalize well without task-specific training.

  ### 5.3 Supervised CNNs are still important baselines

  Although CLIP performs better in this experiment, ResNet18 and ResNet50 remain important baselines.

  They provide a controlled supervised learning comparison and help answer whether the target dataset itself is sufficient for training a scene classifier.

  For robotics, supervised CNNs may still be useful when:

  - the target environment has domain-specific visual patterns
  - labels are fixed and well-defined
  - inference speed and deployment simplicity are important
  - the model needs to be fine-tuned to a specific robot camera or environment

  ## 6. Interpretation for Robot Scene Understanding

  For robot navigation and embodied AI, this comparison suggests the following:

  1. Traditional CNN fine-tuning can learn useful indoor scene features, but its performance depends heavily on the size and diversity of the training dataset.
  2. Vision-language models such as CLIP provide stronger semantic priors, especially when the task involves human-level scene concepts.
  3. For semantic navigation, CLIP-like models are useful because they can connect visual observations with language goals.
  4. For real robot deployment, a hybrid system may be more practical:
      - CNN or lightweight model for fast local perception
      - CLIP/VLM for open-vocabulary semantic understanding
      - traditional navigation stack for geometry, safety, and control

  ## 7. Limitations

  This experiment is still a basic comparison. Several limitations remain:

  - Only top-1 accuracy is compared.
  - No per-class accuracy analysis has been included yet.
  - No detailed confusion matrix comparison between ResNet and CLIP has been done.
  - CLIP prompt engineering has not been tested.
  - No real robot first-person camera data has been evaluated.
  - The experiment does not yet test navigation performance, only image-level scene recognition.

  ## 8. Next Steps

  Recommended follow-up experiments:

  1. Compare per-class accuracy - of {class_name}
      - an indoor scene of a {class_name}

  - a robot view of a {class_name}

  5. Evaluate the models on robot first-person images or navigation logs.
  6. Extend the task from scene classification to navigation failure recognition.

  ## 9. Current Conclusion

  In this experiment, ResNet50 improves over ResNet18, but CLIP zero-shot achieves the best accuracy.

  The result supports the idea that vision-language pretraining is highly valuable for robot scene understanding. For future semantic navigation or navigation failure diagnosis, CLIP/VLM-based methods are promising starting points, while supervised CNNs remain useful
  engineering baselines.
