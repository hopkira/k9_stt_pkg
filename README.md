# K9 Jetson GPU Speech to Text replacement — whisper.cpp

This bundle replaces the Pi-based Python faster-whisper STT node with a C++ ROS 2 node linked directly to the CUDA-enabled whisper.cpp build already proven on the
Jetson Orin NX. The STT executable is native C++ in a new `k9_stt_pkg` ament_cmake package. The large-v3-turbo STT model is loaded once with `use_gpu=true` through libwhisper. In addition, whisper.cpp's Silero Voice Activity Detection (VAD) replaces Python WebRTC VAD. The VAD runs on CPU; Whisper runs on the Orin GPU.

## What it does
This node simply listens to the audio frames captured and published by the Hotword Node and transcribes what it hears into discrete sentences for further processing.

- HotwordNode permanently owns the CM108 microphone.
- `/audio/raw` remains `k9_interfaces_pkg/msg/AudioFrame` at 16 kHz mono.
- `/audio/effective_state == LISTENING` enables STT.
- `/speech_to_text/text` remains the command output.
- One LISTENING period produces at most one command.
- Results become stale when the BT leaves LISTENING.

## Testing the node

Hotword must be running first; check that it is publishing audio:

```bash
ros2 run k9_system_pkg hotword  --ros-args  --params-file ~/k9_ws/src/k9_system_pkg/config/hotword.yaml
ros2 topic hz /audio/raw
```

Then:

```bash
ros2 launch k9_stt_pkg stt.launch.py
```

In another terminal:

```bash
ros2 topic echo /speech_to_text/state
ros2 topic echo /speech_to_text/text
```

Force listening:

```bash
ros2 topic pub --once /audio/effective_state std_msgs/msg/String \
  "{data: 'LISTENING'}"
```

Speak one sentence, then stop. Expected state sequence:

```text
listening
speech
transcribing
result
```

## Configuration options

The default configuration options are quite aggressive, but seem to work well:

```yaml
silence_timeout_sec: 0.4
pre_roll_ms: 200
min_speech_ms: 160
beam_size: 1
flash_attn: true
```
