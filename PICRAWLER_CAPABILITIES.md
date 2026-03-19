# PiCrawler Robot — Complete Hardware & Software Capabilities

## 1. Physical Specifications

- **Type**: Quadruped spider robot (4 legs)
- **Servos**: 12 total (3 per leg: shoulder, thigh, hip)
- **Servo Angle Limits**:
  - Alpha (shoulder): -90° to +90°
  - Beta (thigh): -10° to +90°
  - Gamma (hip rotation): -60° to +60°
- **Leg Dimensions**: Upper=48mm, Middle=78mm, Lower=33mm, Body=77mm
- **Speed Range**: 0-100 (0=stop, 100=fastest)
- **Max Servo Speed**: 428 degrees/second

---

## 2. Movement Actions (via `picrawler.do_action`)

### Basic Locomotion

| Action Name | Description |
|-------------|-------------|
| `forward` | Walk forward |
| `backward` | Walk backward |
| `turn left` | Rotate left in place |
| `turn right` | Rotate right in place |
| `turn left angle` | Rotate left with body tilting (30° default) |
| `turn right angle` | Rotate right with body tilting (30° default) |

### Posture (via `do_action` or `do_step`)

| Action Name | Description |
|-------------|-------------|
| `stand` | Stand up from sitting |
| `sit` | Sit down (resting position) |
| `ready` | Ready state before movement |

### Gestures

| Action Name | Description |
|-------------|-------------|
| `wave` | Raise front-left leg and wave |
| `push up` | Lower and raise the body |
| `dance` | Full body dance with circular movements and spirals |
| `look up` | Tilt body upward |
| `look down` | Tilt body downward |
| `look left` | Tilt body left (30° angle) |
| `look right` | Tilt body right (30° angle) |

### Advanced Actions (from examples)

| Action Name | Description |
|-------------|-------------|
| `fighting` | Combat readiness + twist + pounce |
| `excited` | Vertical bouncing |
| `shake_hand` | Extend front leg for handshake |
| `bull_fight` | Bull fighting pose |

### Custom Actions

- `add_action(action_name, action_list)` — Register any new action as a sequence of coordinate steps

---

## 3. Low-Level Control

### Single Leg Control

- `do_single_leg(leg_index, [x, y, z], speed)` — Move one leg independently
  - Leg 0: Right front
  - Leg 1: Left front
  - Leg 2: Left rear
  - Leg 3: Right rear

### Body Manipulation

- `mix_step(basic_step, leg, coordinate)` — Modify one leg in a full body step
- Coordinate system per leg: `[X, Y, Z]`
  - X: 40-80mm (forward/backward from shoulder)
  - Y: -20 to +20mm (left/right lateral)
  - Z: -50 to -10mm (height, negative is down)
- Default standing position: `[60, 0, -30]` per leg

### Servo-Level

- `set_angle(angles_list, speed)` — Set all 12 servo angles directly
- `coord2polar(coord)` — Convert XYZ coordinates to servo angles (alpha, beta, gamma)
- `polar2coord(angles)` — Convert servo angles to XYZ coordinates

---

## 4. Sensors

### Ultrasonic Distance Sensor

- **Module**: `robot_hat.Ultrasonic`
- **Pins**: D2 (trigger), D3 (echo)
- **Range**: ~2-400cm
- **Method**: `read(times=10)` — Returns distance in cm (median of N readings)
- **Returns**: -1 on timeout, -2 on error
- **Timeout**: 20ms default

### ADC (Analog-to-Digital Converter)

- **Module**: `robot_hat.ADC`
- **Channels**: A0-A7 (8 channels)
- **Resolution**: 12-bit (0-4095)
- **Reference Voltage**: 3.3V
- **Methods**:
  - `read()` — Raw value (0-4095)
  - `read_voltage()` — Voltage (0-3.3V)

### Accelerometer (ADXL345)

- **Module**: `robot_hat.Accelerometer`
- **Address**: 0x53 (I2C)
- **Axes**: X, Y, Z
- **Method**: `read(axis)` — Returns acceleration in g-force

---

## 5. Camera & Vision (Vilib)

### Camera Control

| Method | Description |
|--------|-------------|
| `camera_start(vflip, hflip, size)` | Start camera (default 640x480, 60 FPS) |
| `camera_close()` | Stop camera |
| `display(local, web)` | Show output locally and/or via web (port 9000) |
| `take_photo(name, path)` | Capture photo as JPEG |
| `rec_video_start()` | Start video recording (AVI, 30 FPS) |
| `rec_video_stop()` | Stop video recording |
| `show_fps()` / `hide_fps()` | Toggle FPS display |

### Web Endpoints (port 9000)

| Endpoint | Description |
|----------|-------------|
| `/mjpg` | MJPEG video stream |
| `/mjpg.jpg` | Single JPEG frame |
| `/mjpg.png` | Single PNG frame |

### Face Detection

- **Method**: `face_detect_switch(flag)` — Enable/disable
- **Model**: OpenCV Haar Cascade (`haarcascade_frontalface_default.xml`)
- **Output**:
  - `face_obj_parameter['x']` — Face center X
  - `face_obj_parameter['y']` — Face center Y
  - `face_obj_parameter['w']` — Face width (pixels)
  - `face_obj_parameter['h']` — Face height (pixels)
  - `face_obj_parameter['n']` — Number of faces detected

### Color Detection

- **Method**: `color_detect(color)` — Enable for specific color
- **Supported Colors**: `red`, `orange`, `yellow`, `green`, `blue`, `purple`, `magenta`
- **Method**: `close_color_detection()` — Disable
- **Output**:
  - `color_obj_parameter['x']` — Largest block center X
  - `color_obj_parameter['y']` — Largest block center Y
  - `color_obj_parameter['w']` — Block width
  - `color_obj_parameter['h']` — Block height
  - `color_obj_parameter['n']` — Number of blocks
  - `color_obj_parameter['color']` — Color name

### Hand Detection

- **Method**: `hands_detect_switch(flag)` — Enable/disable
- **Framework**: MediaPipe
- **Output**: `detect_obj_parameter['hands_joints']` — 21 keypoints per hand `[[x, y, z], ...]`
- **Confidence**: 50% detection, 50% tracking

### Pose Detection

- **Method**: `pose_detect_switch(flag)` — Enable/disable
- **Framework**: MediaPipe
- **Output**: `detect_obj_parameter['body_joints']` — 33 body landmarks `[[x, y, z], ...]`
- **Landmarks include**: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles

### Object Detection (TFLite)

- **Method**: `object_detect_switch(flag)` — Enable/disable
- **Model**: TensorFlow Lite (COCO detection, `/opt/vilib/detect.tflite`)
- **Labels**: `/opt/vilib/coco_labels.txt` (80 classes)
- **Custom Model**: `object_detect_set_model(path)` / `object_detect_set_labels(path)`
- **Output**: `object_detection_list_parameter` — List of `{'class_id', 'class_name', 'score', 'bounding_box'}`

### Image Classification (TFLite)

- **Method**: `image_classify_switch(flag)` — Enable/disable
- **Custom Model**: `image_classify_set_model(path)` / `image_classify_set_labels(path)`
- **Output**: `image_classification_obj_parameter`

### Traffic Sign Detection

- **Method**: `traffic_detect_switch(flag)` — Enable/disable
- **Output**:
  - `detect_obj_parameter['traffic_sign_x/y/w/h']` — Position and size
  - `detect_obj_parameter['traffic_sign_t']` — Sign type
  - `detect_obj_parameter['traffic_sign_acc']` — Confidence

### QR Code

- **Detection**: `qrcode_detect_switch(flag)` — Enable/disable
- **Output**:
  - `detect_obj_parameter['qr_data']` — Decoded text
  - `detect_obj_parameter['qr_x/y/w/h']` — Position and size
  - `detect_obj_parameter['qr_list']` — All detected QR codes
- **Generation**: `make_qrcode(data, path, version, box_size, border, fill_color, back_color)`

---

## 6. Audio

### Text-to-Speech (TTS)

- **Module**: `robot_hat.TTS`
- **Engines**: `PICO2WAVE` (default, high quality), `ESPEAK`, `ESPEAK_NG`
- **Languages**: en-US, en-GB, de-DE, es-ES, fr-FR, it-IT
- **Methods**:
  - `say(words)` — Speak text
  - `lang(language)` — Set language
  - `supported_lang()` — List languages
- **Espeak Parameters**: amplitude (0-200), speed (80-260 wpm), gap, pitch (0-99)

### Sound Effects & Music

- **Module**: `robot_hat.Music`
- **Methods**:
  - `sound_play(filename, volume)` — Play sound effect
  - `sound_play_threading(filename, volume)` — Play in background
  - `music_play(filename, loops, start, volume)` — Play music
  - `music_set_volume(value)` — Volume 0-100
  - `music_stop()` / `music_pause()` / `music_resume()` — Playback control
  - `play_tone_for(freq, duration)` — Play tone at frequency
- **Musical Notes**: Full range A0-C8 (88 keys, MIDI compatible)
- **Time Signatures**: Configurable (4/4, 3/4, etc.)
- **Key Signatures**: All major keys supported
- **Tempo**: Configurable BPM (default 120)

---

## 7. GPIO & Low-Level Hardware

### Digital Pins

- **Available**: D0-D16 + named pins (SW, USER, LED, RST, BLEINT, BLERST, MCURST, CE)
- **Modes**: OUTPUT, INPUT
- **Pull**: PULL_UP, PULL_DOWN, PULL_NONE
- **Interrupts**: IRQ_FALLING, IRQ_RISING, IRQ_RISING_FALLING
- **Methods**: `value()`, `on()`/`off()`, `irq(handler, trigger, bouncetime)`

### PWM

- **Channels**: P0-P19 (20 channels)
- **Clock**: 72MHz
- **Methods**: `freq()`, `pulse_width()`, `pulse_width_percent()`

### Other Modules

| Module | Description |
|--------|-------------|
| `RGB_LED` | RGB LED control (common anode/cathode), hex/tuple color |
| `Buzzer` | Passive (tone) or Active (on/off) buzzer |
| `Grayscale` | 3-channel line-following sensor (LEFT, MIDDLE, RIGHT) |
| `Motor` / `Motors` | DC motor control, -100 to +100 speed |

---

## 8. Calibration & Configuration

- **Servo Offset File**: `/opt/picrawler/picrawler.config` — 12 servo offsets (-20° to +20°)
- **Robot HAT Config**: `~/.config/robot-hat/robot-hat.conf`
- **Speed Formula**: `motion_time_ms = -9.9 * speed + 1000`
  - Speed 50 → 505ms per frame
  - Speed 100 → 10ms per frame
- **Servo Interpolation**: Linear, 10ms steps, 0.15s init delay
