# Robot Vision with a CNN and an e-puck in Webots

Applied Robotics & AI, Project Part 2 (Assignments 3, 4 and 5).

The project has two parts that work together:

1. **A CNN for image classification** (Google Colab, TensorFlow/Keras), trained on CIFAR-10.
2. **A Webots simulation** in which an e-puck robot finds coloured blocks with HSV colour detection (OpenCV) and photographs CIFAR-10 objects with its camera, so that the CNN can classify them.

## Repository structure

```
robot-vision-part2/
├── README.md                       # this file
├── colab/
│   └── CIFAR10_CNN_Part2.ipynb     # CNN notebook (run in Google Colab)
└── webots/
    ├── README.md                   # detailed guide to the world and the controller
    ├── worlds/
    │   └── robot_vision.wbt        # arena, 3 colour blocks, airplane, car, truck, horse, e-puck
    └── controllers/
        └── robot_vision_controller/
            └── robot_vision_controller.py
```

## Part 1: CNN (Colab notebook)

| Section | Content |
|---|---|
| 1–2 | Setup (trained models and the dataset are kept in Google Drive) and the CIFAR-10 data |
| 3 | Baseline CNN (3 convolution layers) |
| 3b | Optimiser comparison: SGD vs Adam from the same starting weights |
| 4–6 | Improved CNN (VGG-style blocks, batch normalisation, dropout, data augmentation, early stopping): all layers, training, test accuracy, confusion matrix, error analysis |
| 7 | Predictions on three of my own photos |
| 8 | Predictions on images from the e-puck camera in Webots, with a background test |
| 9 | Optimisation for running on a robot: TensorFlow Lite quantisation, pruning, speed per image |
| 10–11 | Summary of results and download of all figures and models |

**To run:** open the notebook in Google Colab, choose a T4 GPU runtime, then `Runtime → Run all` and allow Google Drive access. Upload three of your own photos in section 7 and the `_photo.png` images from Webots in section 8.

## Part 2: Webots simulation

- **World:** a 2.5 m × 2.5 m arena with red, green and blue blocks (0.3 × 0.3 × 0.1 m) and four objects from CIFAR-10 classes: a model airplane, a car and a truck built from basic shapes, and the Webots horse.
- **Controller:** at the start the e-puck (a Supervisor) places all objects at random. It then finds each block by HSV colour detection, prints `I see <colour>` / `I see <colour> again` with a summary, checks the colours of each block image (mean RGB and histograms), and drives to a spot beside each object to save a side-view photo for the CNN. The run ends with a final report.

**To run:** install Python packages `numpy` and `opencv-python` for the Python used by Webots R2023b, open `webots/worlds/robot_vision.wbt`, press Reset and Play. See [`webots/README.md`](webots/README.md) for the details, settings and tests.

## Tools

- Webots R2023b, Python 3, OpenCV, NumPy
- Google Colab, TensorFlow / Keras
- Trello (project management), Git and GitHub (version control)
