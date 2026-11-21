# CNN for CIFAR-10 in Google Colab (Project Part 2)

`CIFAR10_CNN_Part2.ipynb` trains a CNN that recognises the ten CIFAR-10 classes, tests it on my own photos and on photos taken by the e-puck's camera in Webots, and measures how to make it fast enough for the robot. The report (Sections 3.5–3.12 and 4) explains every step and result.

## Folder contents

```
colab/
├── CIFAR10_CNN_Part2.ipynb        the notebook (sections 1–11)
├── README.md                      this file
├── inputs/
│   ├── my_photos/                 the three photos of my own used in section 7
│   └── webots_photos/             the four photos from the final Webots run (17 Nov) used in section 8
└── outputs/                       saved by section 11 of the final run (17 Nov)
    ├── results_summary.json       every number reported (accuracies, predictions, timings, sizes)
    ├── figures/                   the 16 figures (training curves, confusion matrix, predictions, optimisation…)
    └── models/
        ├── cifar10_cnn.keras              improved CNN, 30 layers, 306,154 parameters, 86.13% test accuracy
        ├── baseline_cnn.keras             baseline CNN, 67.42% test accuracy
        ├── cifar10_dynamic_int8.tflite    TensorFlow Lite int8 model for the robot (315 KB, 1.03 ms per image)
        └── class_names.json               the ten class names in output order
```

## How to run

1. Open the notebook in Google Colab and choose `Runtime → Change runtime type → T4 GPU → Save`.
2. `Runtime → Run all`, and allow access to Google Drive when asked.
   * **First run:** both models are trained (about 15–20 minutes on a T4 GPU) and saved to `MyDrive/robot_vision_cnn/`.
   * **Later runs:** the saved models are loaded and training is skipped (about 5 minutes). Set `TRAIN_AGAIN = True` in the settings cell to train again.
3. Section 7: when **Choose files** appears, select the three photos in `inputs/my_photos/`.
4. Section 8: select the four photos in `inputs/webots_photos/` (or new `NNN_<object>_photo.png` files from `webots/controllers/robot_vision_controller/images/`).
5. Section 11 downloads `assignment3_outputs.zip` with the figures, the models and `results_summary.json`.

Training again gives slightly different numbers because the weights start at random values. The numbers in the report come from the final run on 17 November, saved in `outputs/`.

## Main settings

| Setting | Value | Meaning |
|---|---|---|
| `EPOCHS` | 45 | Maximum passes over the training data (early stopping may end sooner) |
| `BATCH_SIZE` | 128 | Images per training step |
| `VAL_SPLIT` | 5000 | Training images kept aside for validation (45,000 train, 5,000 validation, 10,000 test) |
| `TRAIN_AGAIN` | False | True = train both models again even if saved models exist in Drive |
| `QUICK_TEST` | False | True = 2 epochs on a small subset, only to check that every cell runs |

Optimiser Adam (learning rate 0.001, halved by `ReduceLROnPlateau` after 4 epochs without improvement), `EarlyStopping` (patience 10), L2 weight decay 0.0001, dropout 0.2 / 0.3 / 0.4 in the three blocks and 0.5 before the output layer.

## Main results (final run, 17 November)

| Result | Value |
|---|---|
| Improved CNN, CIFAR-10 test accuracy | 86.13% (baseline 67.42%) |
| SGD vs Adam (baseline, 10 epochs) | 60.03% vs 67.46% |
| My own photos | 3 of 3 correct (airplane 87.9%, bird 98.5%, automobile 99.3%) |
| Webots photos | 3 of 4 correct (horse 91.8%, bird 88.8%, airplane 95.3%; truck predicted as airplane 94.1%) |
| Keras vs TFLite int8 | 32.25 ms vs 1.03 ms per image; 3,700 KB vs 315 KB |
| One-shot pruning, 30% of weights | 69.26% (needs fine-tuning) |
