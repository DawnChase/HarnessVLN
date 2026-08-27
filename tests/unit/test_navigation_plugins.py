from __future__ import annotations

import asyncio

from agents import PassthroughVLNAgent
from envs import DummyNavigationEnvironment
from harness import NavigationHarness, NavigationStack
from memory import DummyLandmarkMemory
from schemas import NavGoal, NavTask
from vln import DummyVLNNavigator


def run(coroutine):
    return asyncio.run(coroutine)


def test_passthrough_runs_complete_vln_jobs_across_goals_without_reset() -> None:
    async def scenario():
        goals = (
            NavGoal("goal-1", "go to the far marker"),
            NavGoal("goal-2", "return to the near marker"),
        )
        task = NavTask("compound-1", goals[0], public={"goal_count": 2})
        environment = DummyNavigationEnvironment(goals, targets=(3, 1))
        navigator = DummyVLNNavigator(inference_period_s=0)
        result = await NavigationHarness(timeout_s=1).run_task(
            task,
            NavigationStack(
                PassthroughVLNAgent(poll_period_s=0), environment, vln=navigator
            ),
        )

        assert result.terminal.status == "completed"
        assert result.environment["position"] == 1
        assert result.environment["start_count"] == 1
        assert result.environment["goal_transitions"] == 1
        assert [event.name for event in result.audit].count("vln.navigate.start") == 2
        assert any(
            event.actor == "vln" and event.name == "nav.move.discrete"
            for event in result.audit
        )
        assert result.audit[-1].name == "nav.stop"

    run(scenario())


def test_landmarks_persist_across_new_task_and_memory_instance(tmp_path) -> None:
    class MemoryAgent:
        required_tools = frozenset({"spatial.search", "spatial.remember"})

        def __init__(self, remember=None, search=""):
            self.remember = remember
            self.search = search
            self.found = []

        async def run(self, context):
            if self.search:
                self.found = await context.spatial.search(
                    self.search,
                    frame="map",
                    near_pose=[1.0, 0.0],
                    top_k=5,
                )
            if self.remember:
                await context.spatial.remember(*self.remember)
            await context.nav.stop("completed", "memory operation complete")

    async def execute(task_id, agent, writeback=True):
        goal = NavGoal(f"{task_id}-goal", "memory test")
        return await NavigationHarness(timeout_s=1).run_task(
            NavTask(task_id, goal),
            NavigationStack(
                agent,
                DummyNavigationEnvironment((goal,), targets=(0,)),
                memory=DummyLandmarkMemory(tmp_path, writeback=writeback),
            ),
        )

    async def scenario():
        await execute(
            "task-a",
            MemoryAgent(remember=("kitchen doorway", "map", [2.0, 0.0])),
        )
        reader = MemoryAgent(search="kitchen")
        await execute("task-b", reader)
        assert len(reader.found) == 1
        assert reader.found[0]["source_task_id"] == "task-a"
        assert reader.found[0]["pose"] == [2.0, 0.0]

    run(scenario())


def test_writeback_false_does_not_change_existing_file(tmp_path) -> None:
    class RememberAgent:
        required_tools = frozenset({"spatial.remember"})

        async def run(self, context):
            await context.spatial.remember("temporary", "map", [0.0, 0.0])
            await context.nav.stop("completed")

    async def scenario():
        path = tmp_path / "landmarks.json"
        path.write_text("[]\n")
        before = path.read_bytes()
        goal = NavGoal("goal", "remember")
        await NavigationHarness(timeout_s=1).run_task(
            NavTask("readonly", goal),
            NavigationStack(
                RememberAgent(),
                DummyNavigationEnvironment((goal,), targets=(0,)),
                memory=DummyLandmarkMemory(tmp_path, writeback=False),
            ),
        )
        assert path.read_bytes() == before

    run(scenario())


def test_vln_cancel_is_idempotent_and_does_not_end_task() -> None:
    class CancelAgent:
        required_tools = frozenset(
            {
                "vln.navigate.start",
                "vln.navigate.status",
                "vln.navigate.cancel",
            }
        )

        async def run(self, context):
            job_id = await context.vln.start("keep moving")
            first = await context.vln.cancel(job_id)
            second = await context.vln.cancel(job_id)
            assert first["state"] == second["state"] == "cancelled"
            await context.nav.stop("completed", "cancel semantics checked")

    async def scenario():
        goal = NavGoal("goal", "cancel")
        result = await NavigationHarness(timeout_s=1).run_task(
            NavTask("cancel", goal),
            NavigationStack(
                CancelAgent(),
                DummyNavigationEnvironment((goal,), targets=(100,)),
                vln=DummyVLNNavigator(inference_period_s=1),
            ),
        )
        assert result.terminal.status == "completed"
        assert result.terminal.actor == "agent"

    run(scenario())
