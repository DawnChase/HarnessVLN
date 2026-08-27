from __future__ import annotations

import asyncio
import os
import sys

from envs.dummy import DummyNavigationEnvironment
from schemas import EnvironmentEpisode


class DeviceRecordingEnvironment(DummyNavigationEnvironment):
    async def start(self, task, output):
        self.cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        print("fixture Habitat scene switch stdout", flush=True)
        print("fixture Habitat scene switch stderr", file=sys.stderr, flush=True)
        await asyncio.sleep(0.05)
        return await super().start(task, output)

    def result(self):
        return {
            **super().result(),
            "cuda_visible_devices": self.cuda_visible_devices,
            "worker_pid": os.getpid(),
        }


def from_episode(episode: EnvironmentEpisode) -> DeviceRecordingEnvironment:
    return DeviceRecordingEnvironment(
        episode.setup.get("goal_stream", (episode.task.goal,)),
        targets=episode.setup.get("targets", (0,)),
    )
