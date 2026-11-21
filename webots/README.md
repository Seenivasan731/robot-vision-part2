# Robot Vision with an e-puck in Webots: Applied Robotics & AI Project, Part 2

An e-puck robot places three coloured blocks at random positions in a Webots arena that also holds four objects for the CNN at fixed places (a model airplane, a bird, a truck and a horse: CIFAR-10 classes airplane, bird, truck and horse). It finds the blocks with **HSV colour detection (OpenCV)**, reports every encounter with a summary and a saved camera image, measures the colours of each block image with **OpenCV** (mask → bitwise AND → mean RGB → histogram), and **plans a path to photograph each object** so the images can be classified by the CIFAR-10 CNN in the Colab notebook. Everything runs automatically. The run ends with a final report naming anything that was missed.

## Files

```
robot_vision_part2/
├── worlds/
│   └── robot_vision.wbt                    # arena, 3 blocks, airplane, bird, truck, horse, e-puck
├── protos/                                 # the hand-built objects, so the world file stays short
│   ├── ColourBlock.proto                   # 0.3 x 0.3 x 0.1 m block with physics, colour as a field
│   ├── ModelAirplane.proto
│   ├── ModelBird.proto
│   └── ModelTruck.proto
├── controllers/
│   └── robot_vision_controller/
│       ├── robot_vision_controller.py      # the controller (under 500 lines)
│       └── images/                         # images + block_colours.csv (emptied at each start)
└── README.md
```

## Requirements

| Item | Version used |
|---|---|
| Webots | R2023b |
| Python | 3.13 (*Tools → Preferences → Python command*) |
| Python packages | `numpy`, `opencv-python` |

```
"C:\Users\<user>\AppData\Local\Programs\Python\Python313\python.exe" -m pip install numpy opencv-python
```

## How to run

1. Copy `robot_vision_part2` into `Documents\Webots\` (keep the `protos` folder next to `worlds`).
2. Open `worlds/robot_vision.wbt` and press **Reset** (⏮), then **Play** (▶).
3. Watch the console:
   * the random position of each block;
   * `I see red` on the first encounter of a block, `I see red again` on later ones, each followed by the OpenCV colour check (`Colour check: 2850 red pixels, mean RGB = (243, 34, 37)`) and a summary; the camera image and the filtered image are saved;
   * `Object in view: airplane at 0.80 m - photo saved for the CNN`;
   * `Task complete` once all blocks are found and every object is photographed (in simulation tests: median 59 s), or `Time limit reached (300 s)` with a list of anything missed.
4. The images are in `controllers/robot_vision_controller/images/`. Upload `NNN_airplane_photo.png`, `NNN_bird_photo.png`, `NNN_truck_photo.png` and `NNN_horse_photo.png` (the square crop around each object) to section 8 of the CNN notebook; `NNN_<object>_view.png` is the full camera view, useful for the slides. **Copy the folder somewhere safe before the next run**: each run starts by emptying it.
5. **Reset** and **Play** again for a new random block layout. The objects stay in the same places, so their photos can be repeated and compared.

## World setup

| Element | Details |
|---|---|
| Arena | `RectangleArena`, 2.5 m × 2.5 m, walls 0.1 m high |
| Blocks | `ColourBlock` PROTO nodes (`protos/ColourBlock.proto`, a `Solid`) `DEF RED_BLOCK`, `DEF GREEN_BLOCK`, `DEF BLUE_BLOCK`; box 0.3 × 0.3 × 0.1 m; `boundingObject` (detected by the proximity sensors) and `Physics` (mass 2 kg); matte `PBRAppearance` colours (1 0 0), (0 1 0), (0 0 1) |
| CNN objects | Four CIFAR-10 classes. `DEF OBJ_AIRPLANE`: a model airliner (white fuselage, grey wings and engines, tail fin, on a thin stand; 0.28 m long, 0.155 m high). `DEF OBJ_BIRD`: a robin-like bird perched on a branch (brown body, darker wings and tail, orange breast, yellow-orange beak, black eyes, thin legs; about 0.17 m high). `DEF OBJ_TRUCK`: a truck with an orange cab, a white cargo box and six wheels (0.34 m long, 0.165 m high). These three are built from basic shapes (boxes, cylinders, spheres, a cone, a capsule) as PROTO files in `protos/`, because Webots has no small airplane, bird or truck model; keeping them out of the world file keeps it short (about 75 lines). `DEF OBJ_HORSE`: the Webots `Horse` (scale 0.1, about 0.17 m tall), coloured chestnut brown with a dark mane like most CIFAR-10 horses (the default is white). Fixed places, one in each quarter of the arena: airplane (-0.75, 0.75), bird (0.75, 0.75), truck (0.75, -0.75), horse (-0.75, -0.75), each turned side-on to the centre (a model's front is its +x direction), so one side faces the open middle. Earlier versions used a dog, a cat and a white horse (all classified as "airplane") and then a shiny silver car (also classified as "airplane", because it reflected the blue sky); the objects that are recognised best are those whose real photos look most like simple 3D models. |


| Random placement | Only the blocks are moved. At start-up the controller puts each block at a random position: 0.3 m from the walls, 0.6 m from the robot's start and 0.45 m from the objects and the other blocks, so nothing overlaps (if there is not enough room, only the gap is relaxed). Before that, each object's **photo corridor** (the points 0.4 m and 0.8 m from it towards the centre, on the side the photo is taken from) is marked as taken, so no block can stand in the way of its side-view photo. |
| Robot | `E-puck` at the centre, `supervisor TRUE`, controller `robot_vision_controller`, camera 160 × 120 (`camera_width`, `camera_height`) |

The robot is a **Supervisor** for two reasons: to place the blocks at random, and to know where the CNN objects stand so it can plan where to photograph them. **The blocks are found only by the camera (HSV)**. The objects are *recognised* by the CNN in Colab, not by the controller.

## Hardware notes (e-puck in Webots)

| Device | Used for | Notes |
|---|---|---|
| `camera` | Colour detection, photos | 160 × 120 px (the e-puck default 52 × 39 makes an object only a few pixels wide; 160 × 120 keeps the same 4:3 view with about 3× the detail for the CNN), field of view 0.84 rad, 2.8 cm above the floor, BGRA, no noise |
| `left/right wheel motor` | Driving | Velocity control, max 6.28 rad/s, normal speed 50 %; wheel radius 0.0205 m, wheel distance 0.052 m |
| `ps0`, `ps1` / `ps6`, `ps7` | Front-right / front-left obstacles | Average of each pair > 80 = obstacle |
| `ps3`, `ps4` | Rear obstacles | Stop reversing early |

## How the controller works

### State machine

| State | Behaviour | Ends when |
|---|---|---|
| `SCAN` | One slow full turn on the spot (5.1 s) to look around: at the start, then every 15–20 s | Turn done, or a block or CNN object is seen |
| `SEARCH` | Drive forward | An event below happens |
| `APPROACH_BLOCK` | Steer towards the centre of a colour not found yet | The block is reached |
| `REPORT` | Stop 1 s after `I see <colour>` | → `LEAVE` |
| `LEAVE` | Turn on the spot to move on | → `SEARCH` |
| `PHOTO_TRIP` | Drive to a planned photo spot, then turn to face the object | Photo taken, or 30 s timeout |
| `AVOID` | Reverse 0.6 s (stops early if the rear sensors see something), then turn | → `LEAVE` |

**Priorities each 32 ms step:** report a reached block → photograph an object that is centred → avoid obstacles (front sensor pairs and 0.5 s `TIME_DELAY` turns, plus **stuck detection**: if the wheels drive forward for 2 s but the robot moves less than 1 cm, it backs away. Turning on the spot is not counted, so stopping or turning never triggers it) → approach an unfound colour → photo trip → scan → search.

### Colour detection

* **Region of interest:** only the upper half of the image. The 0.1 m blocks rise above the camera's horizon, the floor never does.
* **Pre-filter:** `cv.medianBlur` (3 × 3) removes single noisy pixels.
* **HSV masks:** `cv.cvtColor(..., cv.COLOR_BGR2HSV)` then `cv.inRange()`:

| Colour | Hue H (0–179) | Saturation S | Value V |
|---|---|---|---|
| Red | 0–5 and 172–179 (red wraps round the hue circle) | ≥ 190 | ≥ 50 |
| Green | 50–70 | ≥ 190 | ≥ 50 |
| Blue | 110–130 | ≥ 190 | ≥ 50 |

  The ranges were set from colours measured in this world. Blocks render at H = 0 / 60 / 120 with S = 210–222, also on shaded faces (shade lowers V, not S). The brown floor has H 4–11 with S ≤ 167, and the walls, sky and trees have S < 160. Of the CNN objects, the white and grey airplane, the brown bird and horse and the truck's white box are well below S = 190; the truck's orange cab and the bird's beak are saturated, but their hues (H ≈ 17–19) are outside all three block ranges (red stops at H = 5). In tests the ranges detected every block pixel and gave no false detections on the floor, walls, sky or animals.
* **Steering threshold:** a colour is followed once it covers 0.3 % of the image (57 pixels at 160 × 120), so the rule scales with the camera size.
* **Encounter rule:** a block is *reached* when its colour fills 40 % of the centre window for 3 frames in a row. It counts *again* only after it has been out of view (< 5 %) for 2 s of searching.

### OpenCV colour check (block analysis)

When a block is reached, `opencv_filter()` checks the colours of the saved image: colour mask with `cv.inRange`, `cv.bitwise_and` to keep only the block, conversion to RGB, the mean R, G, B of the block pixels, and R, G, B histograms (`cv.calcHist`, 16 bins each). The result is printed, the filtered image is saved as `NNN_<colour>_block_filtered.png`, and one row per encounter is added to `block_colours.csv` (time, colour, pixel count, mean R/G/B, 48 histogram bins), ready to plot in the report.

### Object photos (path planning)

For each object not yet photographed, the controller considers photo spots 0.74 m and 0.86 m from it, at ±90° from the direction it faces (a pure side view), with ±75° and ±105° as fallbacks. Photos are only taken from the side (70–110° from the facing direction), because most CIFAR-10 photos are side views (in an earlier version with a dog, a cat and a horse, a front view of the horse at 32 × 32 pixels was only a thin grey shape, which the CNN classified as an airplane). It rejects spots outside the arena, within 0.3 m of another object, or with a block in the line of sight, and drives to the nearest remaining spot. A photo is saved when the object is 0.65–0.92 m away and within ±6° of the image centre. The minimum distance comes from the camera geometry: the camera is 2.8 cm above the floor and sees 0.335 m up for every metre ahead, and a head can stick out 0.13 m towards the camera, so from 0.65 m even the top of the head is in view. Because the object then fills only part of the 160 × 120 image, the controller also saves a **square crop** centred on the object, 0.4 m wide at its distance (pinhole camera model: pixels = focal length × size ÷ distance), so the object fills about 50–75 % of the photo like a CIFAR-10 image. The crop is therefore about 80–110 pixels square, depending on the distance (for example 80 × 80 or 90 × 90), while the full view stays 160 × 120; the notebook resizes every image to 32 × 32 before the CNN sees it. The crop is 0.4 m wide at the object's distance, wider than the longest object (the airplane: 0.28 m long, 0.30 m wingspan), and is centred on the object's direction. Spots that cannot be reached within 30 s are skipped, and tried again later if every spot has been tried.

## Main settings (top of the controller)

| Setting | Default | Meaning |
|---|---|---|
| `HSV_RANGES` | see table | OpenCV HSV limits per colour |
| `REACHED_SHARE`, `CONFIRM_FRAMES`, `GONE_SHARE`, `AGAIN_AFTER` | 0.40, 3, 0.05, 2 s | Encounter rule |
| `TIME_LIMIT` | 300 s | Final report if the task is not complete |
| `DIST_LIMIT`, `TIME_DELAY`, `STUCK_TIME` | 80, 0.5 s, 2 s | Obstacle avoidance |
| `SCAN_EVERY` | 15–20 s | How often the robot looks around |
| `PHOTO_RANGE`, `PHOTO_DISTANCES`, `PHOTO_SIDES`, `SIDE_VIEW` | 0.65–0.92 m; 0.74/0.86 m; ±90°, ±75°, ±105°; 70–110° | Photo distance, planned spots and allowed viewing angle (side views only) |
| `OBJECT_HEIGHT`, `CAMERA_HEIGHT` | 0.175 m, 0.028 m | Used to crop the object photo |

## Output files

| File | When |
|---|---|
| `NNN_<colour>_block.png`, `NNN_<colour>_block_againK.png` | Each block encounter (camera image) |
| `NNN_<colour>_block_filtered.png` | The same image after the colour filter (block only, rest black) |
| `block_colours.csv` | One row per encounter: mean R, G, B and the R, G, B histograms |
| `NNN_<object>_photo.png` | Square crop around each object, for the CNN (one per run: Reset for a new layout and pose) |
| `NNN_<object>_view.png` | The full 160 × 120 camera view when the photo was taken |

## Acknowledgement

The obstacle avoidance and camera handling follow the module's example code `opencv.py` (proximity-sensor pairs `get_left_sensor()` / `get_right_sensor()`, `DIST_LIMIT`, `TIME_DELAY` turns, `move_forward()` / `turn_left()` / `turn_right()`, detection before obstacle avoidance, `cv.imwrite`) and the *Webots with OpenCV* teaching material (`cv.cvtColor` to HSV, `cv.inRange` masks, `cv.bitwise_and`, mean RGB, histograms). New in this project: three colours with measured HSV ranges and a median pre-filter, a state machine with scanning, encounter tracking with summaries, random block placement, rear sensors and stuck detection, planned photo trips to the four CNN objects, and a final report.

## Tests

The controller was tested before delivery with a 2D simulation of the e-puck (wheel kinematics, camera image of the blocks, proximity sensors, walls) over **190 random block layouts** with the objects at their fixed places: every run found all three blocks, photographed all four objects and ended with `Task complete` (median 59 s, maximum 91 s, with the 160 × 120 camera; all object photos were side views between 71° and 110°, median 90°, taken 0.67–0.92 m away). No stuck messages (an earlier stuck check gave 357 false ones in 95 runs while turning). Edge cases were also checked: a block or CNN object missing from the world, the time-limit report, "again" messages, a block hiding an object, and OpenCV not installed. Results in Webots (final run 17 Nov; other tests 21 Nov):

| ID | Test | Expected result | Result |
|---|---|---|---|
| T1 | No block in view | No colour message | Pass: no colour message (21 Nov) |
| T2 | One block reached | `I see <colour>` once + summary + image | Pass: `I see blue`, summary and images (final run, 17 Nov) |
| T3 | Same block reached again | `I see <colour> again` + summary | Pass: `I see <colour> again` and a summary (21 Nov) |
| T4 | Full run, 5 random block layouts | `Task complete`; record the time | Pass: `Task complete` in all 5 layouts (21 Nov) |
| T5 | Light changed (`TexturedBackgroundLight` luminosity 0.5 and 1.5) | Blocks still found, no false messages | Pass: blocks found at 0.5 and 1.5, no false messages (21 Nov) |
| T6 | Summary check | Counts and times match the messages | Pass: counts and times match the messages (21 Nov) |
| T7 | One block removed from the world | Warning at start; `not in this world` in the summary | Pass: warning at start, `not in this world` in the summary (21 Nov) |
| T8 | Object photos | One side-view photo each of the airplane, the bird, the truck and the horse: whole object in the view, filling the square crop | Pass: all four `_photo.png` saved (final run, 17 Nov) |
| T9 | OpenCV filter | Mean RGB close to the block colour (e.g. red: R high, G and B low); filtered image shows only the block | Pass: blue block, 2,764 pixels, mean RGB (25, 25, 196) |
| T10 | Stuck check | No `Stuck` message after stops, turns or photos; `Stuck` only when pushing against something | Pass: no `Stuck` message (21 Nov) |

## Troubleshooting

| Problem | Fix |
|---|---|
| `No module named 'numpy'` / `'cv2'` | Install the packages for the Python that Webots uses. |
| `Warning: no node with DEF ...` | The DEF name in the world must match exactly. |
| A block is not detected | Check its colour is pure (e.g. `baseColor 1 0 0`, `roughness 1`); adjust `HSV_RANGES`. |
| A photo is not taken | The object may be boxed in by blocks and walls: Reset for a new layout. |
