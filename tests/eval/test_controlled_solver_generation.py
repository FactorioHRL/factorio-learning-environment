"""Regression tests for controlled-solver model configuration, without services."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from inspect_ai.model import ChatMessageUser, GenerateConfig, ModelOutput, get_model

from fle.eval.inspect.integration import scorers
from fle.eval.inspect.integration import solver as controlled

PROGRAM = "```python\nprint(inspect_inventory())\n```"


@pytest.fixture(autouse=True)
def _reset_between_tests():
    """Override the global Factorio reset fixture for these service-free tests."""


@pytest.fixture
def run_controlled_step(monkeypatch):
    """Run the real solver for one step with only Factorio and logging stubbed."""
    monkeypatch.setenv("FLE_VISION", "false")
    pool = SimpleNamespace(
        get_server_allocation=AsyncMock(
            return_value=SimpleNamespace(run_idx=0, api_key=None)
        ),
        release_run_idx=AsyncMock(),
    )
    env = Mock()
    env.step.return_value = ({"score": 1, "flows": {}}, 1, False, False, {})
    env.instance.get_elapsed_ticks.return_value = 60
    monkeypatch.setattr(
        controlled, "get_simple_server_pool", AsyncMock(return_value=pool)
    )
    monkeypatch.setattr(controlled.gym, "make", Mock(return_value=env))
    monkeypatch.setattr(
        controlled, "get_environment_info", Mock(return_value={"id": "test"})
    )
    prompt = Mock()
    prompt.generate_for_agent.return_value = "Control Factorio with Python."
    monkeypatch.setattr(controlled, "SystemPromptGenerator", Mock(return_value=prompt))
    formatter = Mock()
    formatter.return_value.format.return_value = SimpleNamespace(
        raw_str="Inventory: {}"
    )
    formatter.format_flows_compact.return_value = "No production flows."
    monkeypatch.setattr(controlled, "TreeObservationFormatter", formatter)
    monkeypatch.setattr(controlled, "transcript", Mock())
    monkeypatch.setattr(scorers, "apply_intermediate_scoring", AsyncMock())
    trajectory = controlled.TrajectoryData()
    monkeypatch.setattr(controlled, "store_as", Mock(return_value=trajectory))

    async def run(model):
        # Avoid tokenizer downloads; request serialization and generation stay real.
        monkeypatch.setattr(model.api, "count_tokens", AsyncMock(return_value=10))
        configured_model = Mock(return_value=model)
        monkeypatch.setattr(controlled, "get_model", configured_model)
        state = SimpleNamespace(
            metadata={"env_id": "iron_ore_throughput", "trajectory_length": 1},
            messages=[ChatMessageUser(content="Begin task.")],
        )
        async with model:
            result = await controlled.factorio_controlled_solver()(state, None)
        configured_model.assert_called_once_with()
        assert result is state
        assert not trajectory.error
        assert trajectory.total_steps == 1
        assert trajectory.production_score == 1
        env.step.assert_called_once()
        assert env.step.call_args.args[0].code.strip() == "print(inspect_inventory())"
        pool.release_run_idx.assert_awaited_once_with(0)

    return run


@pytest.mark.asyncio
@pytest.mark.parametrize("reasoning_effort", [None, "low", "high"])
@pytest.mark.parametrize("max_tokens", [None, 8192])
async def test_controlled_solver_preserves_model_configuration(
    run_controlled_step, reasoning_effort, max_tokens
):
    """Inspect merges the solver's fallback with the caller's generation settings."""
    received = []

    def respond(messages, tools, tool_choice, config):
        received.append(config)
        return ModelOutput.from_content(model="mockllm", content=PROGRAM)

    config = GenerateConfig(
        reasoning_effort=reasoning_effort, max_tokens=max_tokens, temperature=0.25
    )
    model = get_model(
        "mockllm/configuration-test",
        config=config,
        custom_outputs=respond,
        transforms=["middle-out"],
    )
    original_config = model.config.model_copy(deep=True)
    original_model_args = model.api.model_args.copy()

    await run_controlled_step(model)

    assert len(received) == 1
    assert received[0].reasoning_effort == reasoning_effort
    assert received[0].max_tokens == (max_tokens or 4096)
    assert received[0].temperature == 0.25
    assert model.config == original_config
    assert model.api.model_args == original_model_args


@pytest.mark.asyncio
@pytest.mark.parametrize("responses_api", [False, True])
@pytest.mark.parametrize("max_tokens", [None, 8192])
async def test_controlled_solver_serializes_standard_openai_requests(
    run_controlled_step, responses_api, max_tokens
):
    """Exercise the standard provider through a local HTTP transport, with no API calls."""
    requests = []

    def respond(request):
        requests.append(request)
        if responses_api:
            body = {
                "id": "resp_test",
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "model": "o3-mini",
                "output": [
                    {
                        "id": "msg_test",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {"type": "output_text", "text": PROGRAM, "annotations": []}
                        ],
                    }
                ],
                "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
            }
        else:
            body = {
                "id": "chatcmpl_test",
                "object": "chat.completion",
                "created": 0,
                "model": "o3-mini",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": PROGRAM},
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                },
            }
        return httpx.Response(200, json=body)

    model = get_model(
        "openai/o3-mini",
        api_key="local-test-key",
        base_url="https://openai.invalid/v1",
        config=GenerateConfig(
            reasoning_effort="high",
            reasoning_summary="none",
            max_tokens=max_tokens,
            max_retries=0,
        ),
        responses_api=responses_api,
        streaming=False,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    )

    await run_controlled_step(model)

    assert len(requests) == 1
    body = json.loads(requests[0].content)
    assert body["model"] == "o3-mini"
    assert "transforms" not in body
    if responses_api:
        assert requests[0].url.path == "/v1/responses"
        assert body["reasoning"]["effort"] == "high"
        assert body["max_output_tokens"] == (max_tokens or 4096)
    else:
        assert requests[0].url.path == "/v1/chat/completions"
        assert body["reasoning_effort"] == "high"
        assert body["max_completion_tokens"] == (max_tokens or 4096)
