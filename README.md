<h1 align="center">Factorio Learning Environment</h1>
<p align="center">
  <a href="https://jackhopkins.github.io/factorio-learning-environment/leaderboard">Leaderboard</a> | <a href="https://arxiv.org/abs/2503.09617">Paper</a> | <a href="https://jackhopkins.github.io/factorio-learning-environment/versions/0.3.0.html">Website</a> | <a href="https://jackhopkins.github.io/factorio-learning-environment/sphinx/build/html/">Documentation</a> | <a href="https://discord.gg/zKaV2skewa">Discord (#factorio-learning-env)</a>
</p>

<p align="center">
An open source framework for developing and evaluating LLM agents in the game of <a href="https://factorio.com/">Factorio</a>.
</p>

<p align="center">
<img src="https://github.com/JackHopkins/factorio-learning-environment/raw/main/docs/assets/videos/compressed_sulfuric_acid.webp" width="485" height="364" controls/>
<img src="https://github.com/JackHopkins/factorio-learning-environment/raw/main/docs/assets/videos/compressed_red_science.webp" width="485" height="364" controls/>
</p>
<p align="center"><em>Claude Opus 4.1 Plays Factorio</em></p>

## Quick Links

- [Installation](#installation)
- [Environment](#environment)
- [Contributing](#contributing)

## Installation

### Prerequisites

- Docker
- Python 3.10+
- [Factorio](https://www.factorio.com/) (version 2.0.73 or later), only for optional rendering.

```bash
# Core FLE SDK package
pip install factorio-learning-environment

# With optional features
pip install factorio-learning-environment[eval]      # For running experiments
pip install factorio-learning-environment[mcp]       # For MCP protocol support  
pip install factorio-learning-environment[psql]      # For PostgreSQL support
pip install factorio-learning-environment[eval,mcp,psql]  # All features

# Using uv (recommended)
uv sync
```

### Quickstart

Use the CLI:

```bash
# Activate venv
source .venv/bin/activate

# Start Factorio cluster
fle cluster start

# Run evaluation trajectories (requires [eval] dependencies)
fle eval --config configs/gym_run_config.json
```

### Inspect evaluation configuration

`fle inspect-eval` accepts evaluation defaults from `.env`. Explicit CLI flags
override environment variables; existing process variables override `.env`.
Empty environment values are treated as unset. Without configuration, the
existing evaluation defaults apply, including eight attempts and up to eight
parallel connections.

| Environment variable    | CLI flag              | Description                              |
| ----------------------- | --------------------- | ---------------------------------------- |
| `FLE_MODEL`             | `--model`             | Inspect provider/model identifier        |
| `FLE_ENV_ID`            | `--env-id`            | Single task name                         |
| `FLE_TASKS`             | `--tasks`             | Comma-separated task names               |
| `FLE_TRAJECTORY_LENGTH` | `--trajectory-length` | Steps per trajectory                     |
| `FLE_REASONING_EFFORT`  | `--reasoning-effort`  | `low`, `medium`, or `high`               |
| `FLE_REASONING_TOKENS`  | `--reasoning-tokens`  | Reasoning budget, where supported        |
| `FLE_MAX_TOKENS`        | `--max-tokens`        | Maximum output tokens per response       |
| `FLE_TOKEN_LIMIT`       | `--token-limit`       | Total input and output tokens per sample |

Numeric settings must be positive integers. The sample token limit includes
repeated context and may stop a trajectory before its configured step count.
Provider-specific model support and reasoning options are handled by Inspect.

An example configuration for a single task:

```dotenv
FLE_MODEL=openai/<model-id>
FLE_ENV_ID=steel_plate_throughput
FLE_TRAJECTORY_LENGTH=16
```

Set the provider's API key, such as `OPENAI_API_KEY`, separately. A single-attempt
run with one connection uses:

```bash
fle inspect-eval --epochs 1 --max-connections 1
```

For multiple tasks, replace `FLE_ENV_ID` with
`FLE_TASKS=iron_ore_throughput,iron_plate_throughput`. Setting both environment
selectors is an error unless a task selector or task mode is explicitly supplied
on the command line. Explicit `--env-id`, `--tasks`, `--eval-set`,
`--eval-set-file`, or `--task-type` selections bypass environment task defaults.

List the built-in throughput and unbounded production tasks without starting
Factorio or an evaluation:

```bash
fle inspect-eval --list-tasks
```

## Environment

FLE is an agent evaluation environment built on the game of Factorio, a popular resource management simulation game.

Agents interact with **FLE** by code synthesis through a **REPL** (Read-Eval-Print-Loop) pattern:

1. **Observation**: The agent observes the world through the output streams (stderr/stdout) of their last program.
2. **Action**: The agent generates a Python program to perform their desired action.
3. **Feedback**: The environment executes the program, assigns variables, add classes/functions to the namespace, and provides an output stream.

## Contributing

Join our team and contribute to one of the AI research community's most challenging problems - building open-ended / unsaturateable evals for post-AGI frontier models. If you want to contribute, please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/JackHopkins/factorio-learning-environment)
