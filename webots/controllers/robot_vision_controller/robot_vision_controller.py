"""E-puck robot vision controller (Webots R2023b) - Applied Robotics & AI Project, Part 2.
At the start the robot (a Supervisor) moves the three coloured blocks and four CNN objects to random places.
It then finds each block by HSV colour detection, prints "I see <colour>" (or "... again") and a summary,
saves the image and measures its colours (mean RGB, histograms in block_colours.csv). It also drives beside
each object (airplane, bird, truck, horse) and saves a side-view photo for the CIFAR-10 CNN in the Colab notebook."""
import csv
import glob
import math
import os
import random

import numpy as np
from controller import Supervisor

try:
    import cv2 as cv
except ImportError:
    cv = None

# ------------------------------ Settings ------------------------------
TIME_STEP = 32                    # ms, same as WorldInfo.basicTimeStep
MAX_SPEED = 6.28                  # e-puck maximum wheel speed (rad/s)
CRUISE = 0.5 * MAX_SPEED          # normal driving speed
DIST_LIMIT = 80.0                 # a proximity reading above 80 means an obstacle is close
TIME_DELAY = 0.5                  # s, an avoidance turn lasts at least this long, so the robot turns away
STUCK_TIME = 2.0                  # s with the wheels driving forward but the robot not moving 1 cm = stuck
ARENA_HALF = 1.25                 # half the arena size (2.5 m x 2.5 m)
TIME_LIMIT = 300.0                # s, then the final report is printed
WHEEL_RADIUS, AXLE = 0.0205, 0.052  # m, e-puck wheel radius and wheel distance
SCAN_EVERY = (15, 20)             # s, look around (one slow full turn) this often
SCAN_SPEED = 0.5 * CRUISE         # wheel speed while scanning; one full turn takes:
SCAN_TIME = 2 * math.pi * AXLE / (2 * SCAN_SPEED * WHEEL_RADIUS)

BLOCKS = {"red": "RED_BLOCK", "green": "GREEN_BLOCK", "blue": "BLUE_BLOCK"}
OBJECTS = {"airplane": "OBJ_AIRPLANE", "bird": "OBJ_BIRD", "truck": "OBJ_TRUCK", "horse": "OBJ_HORSE"}  # CIFAR-10
COLOURS = tuple(BLOCKS)

# HSV colour ranges for OpenCV (hue H 0-179, saturation S and brightness V 0-255), measured in this
# world. The blocks have H = 0 (red), 60 (green) or 120 (blue) and S 210-222, even on shaded faces.
# Everything else is much less saturated (floor S <= 167; walls, sky and the objects S < 160), so a
# narrow hue window with S >= 190 picks out the blocks and nothing else.
HSV_RANGES = {
    "red":   [((0, 190, 50), (5, 255, 255)), ((172, 190, 50), (179, 255, 255))],  # red wraps round H = 0
    "green": [((50, 190, 50), (70, 255, 255))],
    "blue":  [((110, 190, 50), (130, 255, 255))],
}
REACHED_SHARE = 0.40              # a block is reached when its colour fills 40 % of the centre window
CONFIRM_FRAMES = 3                # ... for 3 frames in a row (ignores a one-frame glimpse)
GONE_SHARE = 0.05                 # below 5 % the block is out of view
AGAIN_AFTER = 2.0                 # s of searching with the block out of view before it can count "again"

PHOTO_RANGE = (0.65, 0.92)        # m, photo distance: from 0.65 m even the head (0.175 m high) is in view
PHOTO_DISTANCES = (0.74, 0.86)    # m, planned photo spot distances from the object
OBJECT_HEIGHT = 0.175             # m, the objects are at most about this tall in the world
CAMERA_HEIGHT, CAMERA_FORWARD = 0.028, 0.03   # m, camera position above the floor / ahead of the centre
PHOTO_SIDES = (90, -90, 75, -75, 105, -105)  # deg from the object's facing direction: side views
PHOTO_AIM = math.radians(6)       # photo when the object is within +/- 6 deg of the centre
CAMERA_HALF_FOV = 0.42            # rad, half the e-puck camera field of view (0.84 rad)
SIDE_VIEW = (math.radians(70), math.radians(110))  # photo only from the side, as in most CIFAR-10 photos
TRIP_TIMEOUT = 30.0               # s, give up a photo spot that cannot be reached
SEARCH_BETWEEN_TRIPS = 10.0       # s of free block search between photo trips
IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")

# ------------------------------ Robot and devices ------------------------------
robot = Supervisor()
me = robot.getSelf()
left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')
for motor in (left_motor, right_motor):
    motor.setPosition(float('inf'))           # velocity control
    motor.setVelocity(0.0)
ps = {name: robot.getDevice(name) for name in ('ps0', 'ps1', 'ps3', 'ps4', 'ps6', 'ps7')}
for sensor in ps.values():                    # ps0/ps1 front right, ps6/ps7 front left, ps3/ps4 rear
    sensor.enable(TIME_STEP)
camera = robot.getDevice('camera')
camera.enable(TIME_STEP)
W, H = camera.getWidth(), camera.getHeight()
# Region of interest: colour is searched for in the upper half of the image only. The blocks (10 cm)
# are taller than the camera (2.8 cm above the floor), so they appear there; the floor never does.
TOP, BOTTOM = int(H * 0.05), int(H * 0.5)
LEFT, RIGHT = int(W * 0.3), int(W * 0.7)      # centre window, used to decide that a block is reached
FOCAL = (W / 2) / math.tan(CAMERA_HALF_FOV)   # focal length in pixels (pinhole camera model)
MIN_PIXELS = max(6, int(0.003 * W * H))       # colour pixels before steering to a block (57)
COLOUR_LOG = os.path.join(IMAGE_DIR, "block_colours.csv")

image_count = 0
blocks = {c: {"count": 0, "first": None, "frames": 0, "at_block": False, "away": 0.0} for c in COLOURS}
photographed = set()                          # objects already photographed for the CNN
present_blocks = [c for c in COLOURS if robot.getFromDef(BLOCKS[c])]      # DEF nodes in this world
present_objects = [a for a in OBJECTS if robot.getFromDef(OBJECTS[a])]


def display_message(text):
    print(text, flush=True)                   # straight to the Webots console


# ------------------------------ Sensors and movement ------------------------------
def get_left_sensor():                        # True if the front-left pair (ps6, ps7) sees an obstacle
    return (ps['ps7'].getValue() + ps['ps6'].getValue()) / 2.0 > DIST_LIMIT


def get_right_sensor():                       # True if the front-right pair (ps0, ps1) sees an obstacle
    return (ps['ps0'].getValue() + ps['ps1'].getValue()) / 2.0 > DIST_LIMIT


def get_rear_sensor():                        # True if the rear pair (ps3, ps4) sees an obstacle
    return (ps['ps3'].getValue() + ps['ps4'].getValue()) / 2.0 > DIST_LIMIT


def set_speed(left, right):
    left_motor.setVelocity(max(-MAX_SPEED, min(MAX_SPEED, left)))
    right_motor.setVelocity(max(-MAX_SPEED, min(MAX_SPEED, right)))


def stop():
    set_speed(0.0, 0.0)


def move_forward():
    set_speed(CRUISE, CRUISE)


def turn_left():
    set_speed(0.1 * MAX_SPEED, 0.5 * MAX_SPEED)


def turn_right():
    set_speed(0.5 * MAX_SPEED, 0.1 * MAX_SPEED)


def spin(direction):                          # turn on the spot: +1 = left, -1 = right
    set_speed(-direction * CRUISE, direction * CRUISE)


# ------------------------------ World setup: random positions ------------------------------
def free_spot(taken, start):
    """Random (x, y): 0.3 m from walls, 0.6 m from the start, 0.45 m (or less) from objects."""
    lim = ARENA_HALF - 0.3
    for gap in (0.45, 0.4, 0.35, 0.3):
        for _ in range(1000):
            x, y = random.uniform(-lim, lim), random.uniform(-lim, lim)
            if (math.hypot(x - start[0], y - start[1]) > 0.6 and
                    all(max(abs(x - tx), abs(y - ty)) > gap for tx, ty in taken)):
                return x, y
    return None


def place_objects():
    """Place the CNN objects, then the blocks, at random free positions, keeping each photo corridor free."""
    start, taken = me.getPosition()[:2], []
    for name, def_name in list(OBJECTS.items()) + list(BLOCKS.items()):
        node = robot.getFromDef(def_name)
        spot = free_spot(taken, start) if node else None
        if spot is None:
            display_message("Warning: %s skipped (%s)" % (name, "no room" if node else "DEF %s missing" % def_name))
            continue
        x, y = spot
        node.getField('translation').setSFVec3f([x, y, node.getField('translation').getSFVec3f()[2]])  # same height
        if name in OBJECTS:   # side-on to the centre (+/- 30 deg); its photo spot and the way to it stay free
            towards = math.atan2(-y, -x) + random.uniform(-0.52, 0.52)
            node.getField('rotation').setSFRotation([0, 0, 1, towards + random.choice((-1, 1)) * math.pi / 2])
            taken += [(x + d * math.cos(towards), y + d * math.sin(towards)) for d in (0.4, 0.8)]
        node.resetPhysics()                   # blocks have physics: start them at rest
        taken.append((x, y))
        display_message("%-5s placed at x = %5.2f, y = %5.2f" % (name, x, y))


# ------------------------------ Vision: HSV colour detection (OpenCV) ------------------------------
def get_camera_bgr():                         # Webots gives BGRA bytes; OpenCV works with BGR arrays
    img = camera.getImage()
    return np.frombuffer(img, dtype=np.uint8).reshape((H, W, 4))[:, :, :3] if img else None


def colour_mask(hsv, colour):
    """Black-and-white mask (255 where the pixel has this colour). Red needs two hue ranges joined."""
    masks = [cv.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8)) for lo, hi in HSV_RANGES[colour]]
    return masks[0] if len(masks) == 1 else cv.bitwise_or(masks[0], masks[1])


def detect_block(hsv, colour):
    """(share of the centre window, horizontal position from -1 = left to +1 = right, pixel count) of a colour."""
    mask = colour_mask(hsv, colour) > 0
    pixels, share = int(mask.sum()), float(mask[:, LEFT:RIGHT].mean())
    if pixels == 0:
        return share, 0.0, 0
    return share, (np.nonzero(mask)[1].mean() - (W - 1) / 2.0) / ((W - 1) / 2.0), pixels


def detect_all_blocks():
    """Check one frame for all colours: a 3 x 3 median blur removes noisy pixels, then one HSV conversion."""
    bgr = get_camera_bgr()
    if bgr is None:
        return {c: (0.0, 0.0, 0) for c in COLOURS}
    hsv = cv.cvtColor(cv.medianBlur(np.ascontiguousarray(bgr[TOP:BOTTOM]), 3), cv.COLOR_BGR2HSV)
    return {c: detect_block(hsv, c) for c in COLOURS}


def opencv_filter(bgr, colour, now):
    """Colour check: block pixels only (mask + bitwise AND), mean R, G, B and 16-bin histograms -> block_colours.csv."""
    mask = colour_mask(cv.cvtColor(bgr, cv.COLOR_BGR2HSV), colour)
    filtered = cv.bitwise_and(bgr, bgr, mask=mask)            # block pixels kept, everything else black
    rgb = cv.cvtColor(filtered, cv.COLOR_BGR2RGB)
    pixels, (r, g, b) = cv.countNonZero(mask), cv.mean(rgb, mask=mask)[:3]       # mean of block pixels only
    hist = [int(v) for i in range(3) for v in cv.calcHist([rgb], [i], mask, [16], [0, 256]).ravel()]
    display_message("  Colour check: %d %s pixels, mean RGB = (%.0f, %.0f, %.0f)" % (pixels, colour, r, g, b))
    with open(COLOUR_LOG, "a", newline="") as f:
        csv.writer(f).writerow([round(now, 1), colour, pixels, round(r), round(g), round(b)] + hist)
    return filtered


def save_image(image, tag):
    """Save an image with cv.imwrite to controllers/<this>/images/NNN_<tag>.png."""
    global image_count
    image_count += 1
    path = os.path.join(IMAGE_DIR, "%03d_%s.png" % (image_count, tag))
    ok = cv.imwrite(path, np.ascontiguousarray(image))
    display_message(("Image saved: " if ok else "Failed to save image: ") + path)


def start_image_folder():
    """Delete the last run's images (so runs never mix) and start block_colours.csv with its header."""
    os.makedirs(IMAGE_DIR, exist_ok=True)
    for old in glob.glob(os.path.join(IMAGE_DIR, "*.png")):
        os.remove(old)
    with open(COLOUR_LOG, "w", newline="") as f:
        csv.writer(f).writerow(["time_s", "colour", "pixels", "mean_R", "mean_G", "mean_B"] +
                               ["%s_bin%d" % (ch, i) for ch in "RGB" for i in range(16)])


# ------------------------------ Blocks: encounters, messages, summaries ------------------------------
def block_reached(colour, share, searching):
    """True only once per visit to a block, not on every frame it stays in view. The block counts again
    only after it has been out of view for AGAIN_AFTER seconds while the robot was searching."""
    b = blocks[colour]
    b["frames"] = b["frames"] + 1 if share >= REACHED_SHARE else 0
    if not b["at_block"]:
        if b["frames"] >= CONFIRM_FRAMES:
            b["at_block"], b["away"] = True, 0.0
            return True
        return False
    b["away"] = 0.0 if share >= GONE_SHARE else b["away"] + (TIME_STEP / 1000.0 if searching else 0.0)
    if b["away"] >= AGAIN_AFTER:              # away long enough: the next visit counts as "again"
        b["at_block"] = False
    return False


def print_summary(now):
    """Summary of the blocks found and the object photos taken so far."""
    display_message(" Summary (t = %.1f s)" % now)
    for c in COLOURS:
        n, status = blocks[c]["count"], "not found yet" if c in present_blocks else "not in this world"
        if n:
            status = "seen %d time%s (first at %.1f s)" % (n, "" if n == 1 else "s", blocks[c]["first"])
        display_message("  %-5s block : %s" % (c.capitalize(), status))
    missing = [c for c in present_blocks if blocks[c]["count"] == 0]
    display_message("  Blocks found: %d of %d - %s" % (len(present_blocks) - len(missing), len(present_blocks),
                    "still looking for " + ", ".join(missing) if missing else "all blocks found"))
    display_message("  Object photos for the CNN: " +
                    ", ".join("%s %s" % (a, "taken" if a in photographed else "not yet") for a in present_objects))
    display_message("-" * 52)


def report_block(colour, now):
    """'I see <colour>' / 'I see <colour> again', save the image, print the summary."""
    b = blocks[colour]
    b["count"] += 1
    again, b["first"] = b["count"] > 1, b["first"] or now     # time of the first encounter
    display_message("I see %s%s" % (colour, " again" if again else ""))
    bgr = get_camera_bgr()
    if bgr is not None:                       # save the camera image and the colour-filtered image
        tag = "%s_block%s" % (colour, "_again%d" % b["count"] if again else "")
        save_image(bgr, tag)
        save_image(opencv_filter(bgr, colour, now), tag + "_filtered")
    print_summary(now)


def final_report(now, complete):
    """End of the run: say what was done and name anything that was missed."""
    display_message("Task complete" if complete else "Time limit reached (%.0f s)" % TIME_LIMIT)
    missed = (["%s block (check DEF %s, HSV range)" % (c, BLOCKS[c]) for c in COLOURS if blocks[c]["count"] == 0] +
              ["%s photo (view blocked or not reached)" % a for a in present_objects if a not in photographed])
    if missed:
        display_message("Missed: " + "; ".join(missed))
    print_summary(now)


# ------------------------------ CNN objects: photos for the CNN ------------------------------
def wrap(angle):                              # angle in the range -pi ... +pi
    return (angle + math.pi) % (2 * math.pi) - math.pi


def robot_pose():                             # robot position (x, y) and heading in radians
    x, y = me.getPosition()[:2]
    r = me.getOrientation()                   # 3 x 3 rotation matrix, row by row
    return x, y, math.atan2(r[3], r[0])       # direction of the robot's forward (x) axis


def view_blocked(x, y, ax, ay):
    """True if a block stands on the line of sight from (x, y) to the object at (ax, ay)."""
    dx, dy = ax - x, ay - y
    for node in filter(None, map(robot.getFromDef, BLOCKS.values())):
        bx, by = node.getPosition()[:2]
        t = max(0.0, min(1.0, ((bx - x) * dx + (by - y) * dy) / (dx * dx + dy * dy)))  # nearest point
        if math.hypot(x + t * dx - bx, y + t * dy - by) < 0.22:   # half block diagonal + margin
            return True
    return False


def object_target():
    """(name, distance, bearing) of an object in view and ready for a photo, or None."""
    x, y, heading = robot_pose()
    for a in [a for a in present_objects if a not in photographed]:
        node = robot.getFromDef(OBJECTS[a])
        ax, ay = node.getPosition()[:2]
        distance = math.hypot(ax - x, ay - y)
        bearing = wrap(math.atan2(ay - y, ax - x) - heading)
        side = math.atan2(y - ay, x - ax)     # the side of the object the robot is looking at
        if (PHOTO_RANGE[0] <= distance <= PHOTO_RANGE[1] and abs(bearing) <= CAMERA_HALF_FOV
                and SIDE_VIEW[0] <= abs(wrap(side - node.getField('rotation').getSFRotation()[3])) <= SIDE_VIEW[1]
                and not view_blocked(x, y, ax, ay)):
            return a, distance, bearing
    return None


def plan_photo_trip(tried):
    """Path planning: try spots beside each object still to be photographed (side views), drop spots
    outside the arena, too close to another object or with a block in the way, return the nearest."""
    x, y, _ = robot_pose()
    others = [n.getPosition()[:2] for n in map(robot.getFromDef, list(BLOCKS.values()) + list(OBJECTS.values())) if n]
    options = []
    for a in [a for a in present_objects if a not in photographed]:
        node = robot.getFromDef(OBJECTS[a])
        ax, ay = node.getPosition()[:2]
        facing = node.getField('rotation').getSFRotation()[3]
        for offset, dist in [(o, d) for o in PHOTO_SIDES for d in PHOTO_DISTANCES]:
            side = facing + math.radians(offset)
            sx, sy = ax + dist * math.cos(side), ay + dist * math.sin(side)
            if ((a, offset, dist) in tried or abs(sx) > ARENA_HALF - 0.15 or abs(sy) > ARENA_HALF - 0.15
                    or any(max(abs(sx - ox), abs(sy - oy)) < 0.3 for ox, oy in others)
                    or view_blocked(sx, sy, ax, ay)):
                continue
            options.append((math.hypot(sx - x, sy - y), a, offset, dist, sx, sy))
    if not options:
        tried.clear()                         # every spot tried: allow them again next time
        return None
    _, a, offset, dist, sx, sy = min(options)
    tried.add((a, offset, dist))
    return a, sx, sy


def drive_photo_trip(trip):
    """Drive to the photo spot, then turn on the spot to face the object."""
    x, y, heading = robot_pose()
    name, gx, gy = trip
    if math.hypot(gx - x, gy - y) > 0.06:     # not there yet: head for the spot
        error = wrap(math.atan2(gy - y, gx - x) - heading)
        forward = CRUISE * max(0.0, math.cos(error))
    else:                                     # at the spot: face the object
        ax, ay = robot.getFromDef(OBJECTS[name]).getPosition()[:2]
        error, forward = wrap(math.atan2(ay - y, ax - x) - heading), 0.0
    turn = max(-1.0, min(1.0, 2 * error)) * 0.4 * CRUISE
    set_speed(forward - turn, forward + turn)


def crop_object(bgr, d, bearing):
    """Square crop (0.4 m wide at the object) centred on it, so it fills the photo like a CIFAR-10 image.
    Pinhole camera: size in pixels = FOCAL x size in metres / d (distance from the camera)."""
    half = min(H // 2, int(FOCAL * 0.2 / d))                      # 0.4 m fits the longest object (truck)
    cx = int(W / 2 - FOCAL * math.tan(bearing))                    # bearing > 0: object left of centre
    cy = int(H / 2 - FOCAL * (OBJECT_HEIGHT / 2 - CAMERA_HEIGHT) / d)   # row of the object's middle
    left, top = max(0, min(W - 2 * half, cx - half)), max(0, min(H - 2 * half, cy - half))
    return bgr[top:top + 2 * half, left:left + 2 * half]


def photograph_object(name, distance, bearing):
    """Stop, let the camera settle, save the full camera view and the cropped photo for the CNN."""
    stop()
    robot.step(TIME_STEP)
    photographed.add(name)
    display_message("Object in view: %s at %.2f m - photo saved for the CNN" % (name, distance))
    bgr = get_camera_bgr()
    if bgr is not None:
        save_image(bgr, "%s_view" % name)
        save_image(crop_object(bgr, distance - CAMERA_FORWARD, bearing), "%s_photo" % name)


# ------------------------------ Main: state machine ------------------------------
def main():
    if cv is None:
        display_message("OpenCV is not installed - run: python -m pip install opencv-python numpy")
        return
    start_image_folder()
    place_objects()
    display_message("Searching for the blocks and photographing the objects...")
    state, until, turn_start = "SCAN", SCAN_TIME, -TIME_DELAY      # start by looking around
    next_scan = SCAN_TIME + random.uniform(*SCAN_EVERY)
    trip, trip_until, next_trip, tried = None, 0.0, SCAN_TIME, set()
    last_xy, last_move = me.getPosition()[:2], 0.0

    while robot.step(TIME_STEP) != -1:
        now = robot.getTime()
        done = all(blocks[c]["count"] > 0 for c in present_blocks) and photographed >= set(present_objects)
        if now >= TIME_LIMIT or done:         # every block found and every object photographed, or out of time
            stop()
            final_report(now, done)
            break
        seen = detect_all_blocks()
        moving = state in ("SEARCH", "SCAN", "APPROACH_BLOCK", "PHOTO_TRIP")

        # 1. REPORT a block that has just been reached (highest priority)
        reached = [c for c in COLOURS if block_reached(c, seen[c][0], moving)]
        if reached:
            for c in reached:
                report_block(c, now)
            stop()
            state, until = "REPORT", now + 1.0
            continue

        # 2. Timed states: REPORT -> LEAVE -> SEARCH, AVOID -> LEAVE, SCAN ends
        if state in ("REPORT", "AVOID"):      # AVOID = reversing away from an obstacle
            if state == "AVOID" and get_rear_sensor():
                until = now                   # something behind: stop reversing
            if now >= until:
                spin(random.choice((-1, 1)))  # then turn away: longer after a block report
                state, until = "LEAVE", now + random.uniform(*((1.3, 1.8) if state == "REPORT" else (0.6, 1.2)))
            continue
        if state == "LEAVE":                  # turning away from a block, obstacle or object
            state = "SEARCH" if now >= until else "LEAVE"
            continue
        if state == "SCAN" and now >= until:  # full turn done
            state = "SEARCH"

        # 3. Photograph an object that is centred and at photo distance
        target = object_target()
        if target and abs(target[2]) <= PHOTO_AIM:
            photograph_object(*target)
            if trip is not None and trip[0] == target[0]:     # this trip's photo is taken
                trip, next_trip = None, now + SEARCH_BETWEEN_TRIPS
            spin(random.choice((-1, 1)))
            state, until = "LEAVE", now + random.uniform(0.8, 1.4)
            continue

        # 4. Obstacle avoidance: turn away from a blocked sensor pair; back up if both are blocked.
        # Stuck check: only while the wheels drive forward (turning on the spot never moves the robot).
        xy = me.getPosition()[:2]
        driving_forward = left_motor.getVelocity() + right_motor.getVelocity() > CRUISE
        if not driving_forward or math.hypot(xy[0] - last_xy[0], xy[1] - last_xy[1]) > 0.01:
            last_xy, last_move = xy, now
        if now - turn_start < TIME_DELAY:
            continue
        left_blocked, right_blocked = get_left_sensor(), get_right_sensor()
        if now - last_move > STUCK_TIME:      # pushing against something the sensors cannot see
            display_message("Stuck - backing away")
            last_move = now
            left_blocked = right_blocked = True
        if left_blocked and right_blocked:
            set_speed(-CRUISE, -CRUISE)       # reverse
            state, until = "AVOID", now + 0.6
            continue
        if left_blocked or right_blocked:     # turn away from the blocked side
            turn_right() if left_blocked else turn_left()
            turn_start, state = now, "SEARCH"
            continue

        # 5. APPROACH_BLOCK: steer towards a colour that has not been found yet
        visible = [c for c in COLOURS if blocks[c]["count"] == 0 and seen[c][2] >= MIN_PIXELS]
        if visible:
            error = seen[max(visible, key=lambda k: seen[k][2])][1]   # -1 left ... +1 right
            set_speed(CRUISE * (1 + 0.6 * error), CRUISE * (1 - 0.6 * error))
            state = "APPROACH_BLOCK"
            continue

        # 6. PHOTO_TRIP: drive to a planned photo spot, then face the object
        if trip is None and now >= next_trip and state != "SCAN":
            trip, trip_until = plan_photo_trip(tried), now + TRIP_TIMEOUT
            next_trip = now + SEARCH_BETWEEN_TRIPS
        if trip is not None and now <= trip_until:
            drive_photo_trip(trip)
            state = "PHOTO_TRIP"
            continue
        trip = None                           # no trip, or the spot was not reached in time: try another later

        # 7. SCAN: one slow full turn to look around (blocks or objects interrupt it)
        if state != "SCAN" and now >= next_scan:
            state, until = "SCAN", now + SCAN_TIME
            next_scan = until + random.uniform(*SCAN_EVERY)
        if state == "SCAN":
            set_speed(-SCAN_SPEED, SCAN_SPEED)
            continue

        # 8. SEARCH: drive forward
        state = "SEARCH"
        move_forward()


if __name__ == '__main__':
    main()
