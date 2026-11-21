# Images from the final Webots run (17 November)

The four object photos saved by the controller and classified by the CNN in notebook section 8 (results in `colab/outputs/results_summary.json`):

| File | CNN prediction |
|---|---|
| `008_horse_photo.png` | horse, 91.8% |
| `010_truck_photo.png` | airplane, 94.1% (wrong; the truck was the one error) |
| `012_bird_photo.png` | bird, 88.8% |
| `016_airplane_photo.png` | airplane, 95.3% |

`007_horse_view.png` and `015_airplane_view.png` are the full 160 × 120 camera views from which the horse and airplane photos were cropped.
Each new run writes its images to `controllers/robot_vision_controller/images/`.
