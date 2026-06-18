# room_guide

Voice-controlled robot navigation using an LLM to interpret natural language
commands and route them through the move_base navigation stack.

The pipeline: wake-word detection, speech-to-text via Whisper, LLM tool-calling
for location matching, and goal dispatch to move_base.

## Nodes

**whisper_node**
Listens for a wake word, records speech, transcribes via Whisper, and publishes
the result to `/llm_input`. Runs as a standalone Python process using its own
virtual environment for ML dependencies.

**llm_nav_node**
Receives text from `/llm_input`, sends it to an LLM with a navigation tool
definition, parses the tool call, and dispatches a `MoveBaseGoal` to the
move_base action server.

**prompt_node**
Provides a text-based terminal interface as an alternative to voice input.
Publishes typed commands to `/llm_input` and prints replies from `/llm_reply`.

**location_recorder_node** (record mode)
Teleoperation interface for saving waypoints. Use `w/a/s/d/x` to drive and `r`
to capture the current AMCL pose to `locations.yaml`.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install numpy sounddevice torch openai-whisper silero_vad openwakeword
```

Wake-word and VAD models are downloaded automatically on first run.

After creating the virtual environment, update the shebang in `scripts/whisper_node.py`
to point at your venv:

```
#!/path/to/room_guide/.venv/bin/python3
```

Place your API key in `config/params.yaml`. This file is gitignored.

## Usage

Navigate mode:

```bash
roslaunch room_guide llm_navigation.launch mode:=navigate
```

Record mode:

```bash
roslaunch room_guide llm_navigation.launch mode:=record
```

With overrides:

```bash
roslaunch room_guide llm_navigation.launch mode:=navigate \
  api_key:="sk-..." model:="openai/gpt-4o" wakeword_name:="hey_computer"
```

## Launch Arguments

`mode`
    Required. `navigate` or `record`.

`api_key`
    OpenRouter or OpenAI-compatible API key.

`model`
    Model identifier passed to the API. Default: `openrouter/free`.

`openai_base`
    Base URL for the chat completions endpoint.
    Default: `https://openrouter.ai/api/v1`.

`input_topic`
    ROS topic the LLM node subscribes to. Default: `/llm_input`.

`navigation_timeout`
    Seconds to wait for a navigation goal before cancelling. Default: `60.0`.

`whisper_model`
    Whisper model size. One of `small`, `medium`, `large`. Default: `small`.

`wakeword_name`
    Wake-word phrase that triggers recording. Default: `hey_jarvis`.

`map_file`
    Path to the map YAML for the navigation stack.
    Default: `$HOME/catkin_ws/maps/RL_1.yaml`.
