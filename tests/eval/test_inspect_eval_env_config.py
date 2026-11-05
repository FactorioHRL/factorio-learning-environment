"""CLI configuration tests without Factorio, Docker, or external API requests."""

import socket
import sys
from unittest.mock import Mock

import pytest
from dotenv import load_dotenv

from fle import run
from fle.eval.inspect.integration import eval_set


@pytest.fixture(autouse=True)
def _reset_between_tests():
    """Override the global Factorio reset fixture for these service-free tests."""


@pytest.fixture
def cli(monkeypatch, tmp_path):
    """Exercise the real parser and command builder with external operations stubbed."""
    monkeypatch.chdir(tmp_path)
    for name in tuple(run.os.environ):
        if name.startswith(("FLE_", "INSPECT_EVAL_")) or name == "OPENAI_API_KEY":
            monkeypatch.delenv(name)
    dotenv = Mock(side_effect=lambda **kwargs: load_dotenv(tmp_path / ".env", **kwargs))
    monkeypatch.setattr(run, "load_dotenv", dotenv)
    initialize = Mock()
    monkeypatch.setattr(run, "fle_init", initialize)
    execute = Mock(return_value=Mock(returncode=0))
    monkeypatch.setattr(run.subprocess, "run", execute)
    monkeypatch.setattr(
        run.subprocess, "Popen", Mock(side_effect=AssertionError("Unexpected viewer"))
    )
    connection = Mock()
    connection.connect_ex.return_value = 0
    monkeypatch.setattr(socket, "socket", Mock(return_value=connection))
    image_exists = Mock(return_value=True)
    build = Mock(side_effect=AssertionError("Unexpected image build"))
    monkeypatch.setattr(run, "_sandbox_image_exists", image_exists)
    monkeypatch.setattr(run, "sandbox_build", build)

    def invoke(*arguments):
        monkeypatch.setattr(sys, "argv", ["fle", "inspect-eval", *arguments])
        run.main()
        return execute.call_args.args[0] if execute.called else None

    invoke.execute = execute
    invoke.dotenv = dotenv
    invoke.initialize = initialize
    invoke.socket = socket.socket
    invoke.image_exists = image_exists
    invoke.build = build
    invoke.path = tmp_path
    return invoke


def option(command, name):
    """Return a generated Inspect option's value."""
    return command[command.index(name) + 1]


def test_dotenv_configures_model_task_reasoning_and_limits(cli):
    (cli.path / ".env").write_text(
        "FLE_MODEL=openai/test-model\n"
        "FLE_ENV_ID=steel_plate_throughput\n"
        "FLE_TRAJECTORY_LENGTH=16\n"
        "FLE_REASONING_EFFORT=high\n"
        "FLE_REASONING_TOKENS=2048\n"
        "FLE_MAX_TOKENS=8192\n"
        "FLE_TOKEN_LIMIT=250000\n"
    )
    command = cli("--epochs", "1", "--max-connections", "1")
    cli.dotenv.assert_called_once_with(override=False)
    assert command[2].endswith("eval_set.py@steel_plate_throughput")
    for flag, value in (
        ("--model", "openai/test-model"),
        ("--reasoning-effort", "high"),
        ("--reasoning-tokens", "2048"),
        ("--max-tokens", "8192"),
        ("--token-limit", "250000"),
        ("--epochs", "1"),
        ("--max-connections", "1"),
    ):
        assert option(command, flag) == value
    assert run.os.environ["FLE_TRAJECTORY_LENGTH"] == "16"
    assert run.os.environ["FLE_ENV_ID"] == "steel_plate_throughput"
    assert "-M" not in command


@pytest.mark.parametrize(
    "variable,flag,environment,explicit",
    [
        ("FLE_MODEL", "--model", "openai/env-model", "openai/cli-model"),
        ("FLE_ENV_ID", "--env-id", "iron_ore_throughput", "iron_plate_throughput"),
        ("FLE_TRAJECTORY_LENGTH", "--trajectory-length", "17", "19"),
        ("FLE_REASONING_EFFORT", "--reasoning-effort", "low", "high"),
        ("FLE_REASONING_TOKENS", "--reasoning-tokens", "1000", "2000"),
        ("FLE_MAX_TOKENS", "--max-tokens", "4000", "8000"),
        ("FLE_TOKEN_LIMIT", "--token-limit", "10000", "20000"),
    ],
)
def test_cli_overrides_environment_and_dotenv(
    cli, monkeypatch, variable, flag, environment, explicit
):
    (cli.path / ".env").write_text(f"{variable}=invalid-dotenv-value\n")
    monkeypatch.setenv(variable, environment)
    command = cli(flag, explicit)
    if variable == "FLE_ENV_ID":
        assert command[2].endswith(f"@{explicit}")
    elif variable == "FLE_TRAJECTORY_LENGTH":
        assert run.os.environ[variable] == explicit
    else:
        assert option(command, flag) == explicit


def test_environment_overrides_dotenv(cli, monkeypatch):
    (cli.path / ".env").write_text("FLE_MODEL=openai/file-model\nFLE_MAX_TOKENS=1000\n")
    monkeypatch.setenv("FLE_MODEL", "openai/env-model")
    monkeypatch.setenv("FLE_MAX_TOKENS", "2000")
    command = cli()
    assert option(command, "--model") == "openai/env-model"
    assert option(command, "--max-tokens") == "2000"


def test_empty_dotenv_values_preserve_existing_defaults(cli):
    variables = (
        "FLE_MODEL",
        "FLE_ENV_ID",
        "FLE_TASKS",
        "FLE_TRAJECTORY_LENGTH",
        "FLE_REASONING_EFFORT",
        "FLE_REASONING_TOKENS",
        "FLE_MAX_TOKENS",
        "FLE_TOKEN_LIMIT",
    )
    (cli.path / ".env").write_text("".join(f'{name}="  "\n' for name in variables))
    command = cli()
    assert command[2].endswith("agent_task.py@factorio_agent_evaluation")
    assert option(command, "--model") == "openai/gpt-4o-mini"
    assert option(command, "--epochs") == "8"
    assert option(command, "--max-connections") == "8"
    assert run.os.environ["FLE_TRAJECTORY_LENGTH"] == "64"
    assert "FLE_ENV_ID" not in run.os.environ
    for flag in (
        "--reasoning-effort",
        "--reasoning-tokens",
        "--max-tokens",
        "--token-limit",
    ):
        assert flag not in command


@pytest.mark.parametrize("source", ["environment", "cli"])
def test_task_list_preserves_order_and_strips_spaces(cli, monkeypatch, source):
    value = " iron_plate_throughput , iron_ore_throughput "
    if source == "environment":
        monkeypatch.setenv("FLE_TASKS", value)
        command = cli()
    else:
        command = cli("--tasks", value)
    assert command[2].endswith("eval_set.py@iron_plate_throughput")
    assert command[3].endswith("eval_set.py@iron_ore_throughput")


def test_environment_task_selectors_conflict_before_side_effects(cli, monkeypatch):
    monkeypatch.setenv("FLE_ENV_ID", "iron_ore_throughput")
    monkeypatch.setenv("FLE_TASKS", "iron_plate_throughput")
    with pytest.raises(SystemExit, match="2"):
        cli()
    cli.initialize.assert_not_called()
    cli.execute.assert_not_called()
    cli.socket.assert_not_called()


@pytest.mark.parametrize(
    "arguments,suffix,operation,steps",
    [
        (
            ["--env-id", "steel_plate_throughput"],
            "eval_set.py@steel_plate_throughput",
            "eval",
            "64",
        ),
        (
            ["--tasks", "steel_plate_throughput"],
            "eval_set.py@steel_plate_throughput",
            "eval",
            "64",
        ),
        (["--eval-set"], "eval_set.py", "eval-set", "64"),
        (["--eval-set-file", "custom.py"], "custom.py", "eval-set", "64"),
        (
            ["--task-type", "unbounded"],
            "eval_set.py@open_play_production",
            "eval",
            "5000",
        ),
        (
            ["--task-type", "throughput"],
            "agent_task.py@factorio_agent_evaluation",
            "eval",
            "64",
        ),
    ],
)
def test_explicit_task_selection_bypasses_both_environment_selectors(
    cli, monkeypatch, arguments, suffix, operation, steps
):
    monkeypatch.setenv("FLE_ENV_ID", "iron_ore_throughput")
    monkeypatch.setenv("FLE_TASKS", "iron_plate_throughput")
    command = cli(*arguments)
    assert command[1] == operation
    assert command[2].endswith(suffix)
    assert run.os.environ["FLE_TRAJECTORY_LENGTH"] == steps
    if "--env-id" not in arguments:
        assert "FLE_ENV_ID" not in run.os.environ


@pytest.mark.parametrize("value", ["0", "-1", "abc", "1.5"])
@pytest.mark.parametrize(
    "variable,flag",
    [
        ("FLE_TRAJECTORY_LENGTH", "--trajectory-length"),
        ("FLE_REASONING_TOKENS", "--reasoning-tokens"),
        ("FLE_MAX_TOKENS", "--max-tokens"),
        ("FLE_TOKEN_LIMIT", "--token-limit"),
    ],
)
@pytest.mark.parametrize("source", ["environment", "cli"])
def test_invalid_numbers_fail_before_side_effects(
    cli, monkeypatch, value, variable, flag, source
):
    if source == "environment":
        monkeypatch.setenv(variable, value)
        arguments = []
    else:
        arguments = [flag, value]
    with pytest.raises(SystemExit, match="2"):
        cli(*arguments)
    cli.initialize.assert_not_called()
    cli.execute.assert_not_called()
    cli.socket.assert_not_called()


@pytest.mark.parametrize(
    "value",
    ["", ",", "iron_ore_throughput,", "iron_ore_throughput,,iron_plate_throughput"],
)
def test_empty_task_list_elements_are_rejected(cli, value):
    with pytest.raises(SystemExit, match="2"):
        cli("--tasks", value)
    cli.execute.assert_not_called()


def test_invalid_environment_reasoning_is_rejected(cli, monkeypatch):
    monkeypatch.setenv("FLE_REASONING_EFFORT", "unknown")
    with pytest.raises(SystemExit, match="2"):
        cli()
    cli.execute.assert_not_called()


def test_list_tasks_has_no_external_or_file_side_effects(cli, monkeypatch, capsys):
    monkeypatch.setenv("FLE_TOKEN_LIMIT", "invalid")
    monkeypatch.setenv("FLE_ENV_ID", "conflicting-default")
    monkeypatch.setenv("FLE_TASKS", "another-default")
    cli("--list-tasks")
    output = capsys.readouterr().out
    for name in (
        *eval_set.ALL_THROUGHPUT_TASKS,
        *eval_set.ALL_UNBOUNDED_PRODUCTION_TASKS,
    ):
        assert name in output
    cli.initialize.assert_not_called()
    cli.dotenv.assert_not_called()
    cli.execute.assert_not_called()
    cli.socket.assert_not_called()
    cli.image_exists.assert_not_called()
    assert not (cli.path / ".env").exists()


def test_sandbox_preserves_environment_task_and_existing_flags(cli, monkeypatch):
    monkeypatch.setenv("FLE_ENV_ID", "iron_ore_throughput")
    command = cli(
        "--sandbox", "--model-base-url", "http://localhost:8000/v1", "--epochs", "1"
    )
    assert command[2].endswith("sandbox_eval_set.py@iron_ore_throughput")
    assert option(command, "--model-base-url") == "http://localhost:8000/v1"
    assert option(command, "--epochs") == "1"
    cli.image_exists.assert_called_once()
    cli.socket.assert_not_called()
    cli.build.assert_not_called()
