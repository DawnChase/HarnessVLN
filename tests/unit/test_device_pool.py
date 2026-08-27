from __future__ import annotations

import asyncio

from harness.process_pool import DevicePool, build_device_slots


def test_device_slots_expand_workers_per_device() -> None:
    slots = build_device_slots(
        {"devices": [2, 5], "workers_per_device": 2}
    )

    assert [(slot.slot_id, slot.physical_device, slot.local_device) for slot in slots] == [
        (0, 2, 0),
        (1, 2, 0),
        (2, 5, 0),
        (3, 5, 0),
    ]


def test_device_pool_leases_requested_slots_as_one_group() -> None:
    async def scenario() -> None:
        pool = DevicePool(build_device_slots({"devices": [3, 7]}))
        entered = asyncio.Event()

        async def wait_for_slot():
            async with pool.lease(1) as leased:
                entered.set()
                return leased

        async with pool.lease(2) as first:
            assert [slot.physical_device for slot in first] == [3, 7]
            waiting = asyncio.create_task(wait_for_slot())
            await asyncio.sleep(0)
            assert not entered.is_set()

        leased = await waiting
        assert leased[0].physical_device == 3

    asyncio.run(scenario())
