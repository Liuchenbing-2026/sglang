"""CPU control-contract tests; scheduler replies are explicit transport fixtures."""

import asyncio
import unittest

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import (
    AbortReq,
    ContinueGenerationReqInput,
    PauseContinueBroadcastReq,
    PauseGenerationReqInput,
)
from sglang.srt.managers.multi_tokenizer_mixin import TokenizerWorker
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.utils.aio_rwlock import RWLock
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


class _ControlTransport:
    """Model the rank boundary without forward, KV allocation, or a service."""

    def __init__(self, worker, mode=DisaggregationMode.NULL):
        cls = TokenizerWorker if worker else TokenizerManager
        self.manager = cls.__new__(cls)
        self.manager.is_pause = False
        self.manager.disaggregation_mode = mode
        self.manager.is_pause_cond = asyncio.Condition()
        self.manager.model_update_lock = RWLock()
        self.manager.enable_metrics = False
        self.manager._dispatch_to_scheduler = self.send
        self.manager._async_dispatch_to_scheduler = self.async_send
        self.worker = worker
        self.paused = False
        self.running = False
        self.messages = []
        self.finished = asyncio.Event()
        self.reader_started = asyncio.Event()
        self.reader_task = None
        self.callbacks = []
        self.fail_retract = False

    def send(self, obj):
        self.messages.append(obj)
        if isinstance(obj, PauseGenerationReqInput):
            if obj.mode == "retract" and self.fail_retract:
                raise RuntimeError("fixture transport rejected retract")
            if obj.mode != "abort":
                self.paused = True
            if obj.mode == "retract":
                self.running = False
            if self.worker:
                self._broadcast(True)
        elif isinstance(obj, ContinueGenerationReqInput):
            self.paused = False
            if self.worker:
                self._broadcast(False)
        elif isinstance(obj, AbortReq):
            # A retained paused running request needs a forward step to finish.
            # Retracted/waiting requests can report an abort without a forward.
            if not self.paused or not self.running:
                self.finished.set()

    async def async_send(self, obj):
        self.send(obj)
        await asyncio.sleep(0)

    def _broadcast(self, paused):
        self.callbacks.append(
            asyncio.create_task(
                self.manager._apply_pause_continue_broadcast(
                    PauseContinueBroadcastReq(is_pause=paused)
                )
            )
        )

    async def start_request(self):
        async def request_reader():
            async with self.manager.model_update_lock.reader_lock:
                self.running = True
                self.reader_started.set()
                await self.finished.wait()
                self.running = False

        self.reader_task = asyncio.create_task(request_reader())
        await self.reader_started.wait()

    async def close(self):
        if self.reader_task is not None:
            self.reader_task.cancel()
            await asyncio.gather(self.reader_task, return_exceptions=True)
        await asyncio.gather(*self.callbacks, return_exceptions=True)


class TestTokenizerPauseAbort(CustomTestCase):
    async def _check_pause_then_abort(self, worker, first_mode):
        transport = _ControlTransport(worker)
        manager = transport.manager
        pause_task = None
        gate_task = None
        try:
            await transport.start_request()
            await manager.pause_generation(PauseGenerationReqInput(mode=first_mode))
            self.assertTrue(manager.is_pause)
            self.assertTrue(transport.paused)
            self.assertTrue(await manager.model_update_lock.is_locked())

            pause_task = asyncio.create_task(
                manager.pause_generation(PauseGenerationReqInput(mode="abort"))
            )
            done, _ = await asyncio.wait({pause_task}, timeout=2)
            self.assertIn(
                pause_task,
                done,
                "abort cannot drain the reader retained by an in-place pause",
            )
            await pause_task
            self.assertFalse(await manager.model_update_lock.is_locked())
            self.assertTrue(manager.is_pause)
            self.assertTrue(transport.paused)
            self.assertFalse(
                any(
                    isinstance(obj, ContinueGenerationReqInput)
                    for obj in transport.messages
                )
            )

            async def wait_until_resumed():
                async with manager.is_pause_cond:
                    await manager.is_pause_cond.wait_for(lambda: not manager.is_pause)

            gate_task = asyncio.create_task(wait_until_resumed())
            await asyncio.sleep(0)
            self.assertFalse(gate_task.done())
            await manager.continue_generation(
                ContinueGenerationReqInput(torch_empty_cache=False)
            )
            await asyncio.wait_for(gate_task, timeout=1)
            self.assertFalse(manager.is_pause)
            self.assertFalse(transport.paused)
        finally:
            for task in (pause_task, gate_task):
                if task is not None and not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
            await transport.close()

    def test_single_worker_in_place_then_abort_releases_reader(self):
        asyncio.run(self._check_pause_then_abort(False, "in_place"))

    def test_tokenizer_worker_in_place_then_abort_releases_reader(self):
        asyncio.run(self._check_pause_then_abort(True, "in_place"))

    def test_retract_then_abort_still_drains(self):
        for worker in (False, True):
            with self.subTest(worker=worker):
                asyncio.run(self._check_pause_then_abort(worker, "retract"))

    def test_initial_abort_does_not_retract_or_pause_scheduler(self):
        async def check(worker):
            transport = _ControlTransport(worker)
            try:
                await transport.start_request()
                await asyncio.wait_for(
                    transport.manager.pause_generation(
                        PauseGenerationReqInput(mode="abort")
                    ),
                    timeout=2,
                )
                self.assertFalse(await transport.manager.model_update_lock.is_locked())
                self.assertFalse(transport.paused)
                self.assertTrue(transport.manager.is_pause)
                self.assertFalse(
                    any(
                        isinstance(obj, PauseGenerationReqInput)
                        and obj.mode == "retract"
                        for obj in transport.messages
                    )
                )
            finally:
                await transport.close()

        for worker in (False, True):
            with self.subTest(worker=worker):
                asyncio.run(check(worker))

    def test_idle_in_place_then_abort_still_completes(self):
        async def check(worker):
            transport = _ControlTransport(worker)
            try:
                await transport.manager.pause_generation(
                    PauseGenerationReqInput(mode="in_place")
                )
                await asyncio.wait_for(
                    transport.manager.pause_generation(
                        PauseGenerationReqInput(mode="abort")
                    ),
                    timeout=1,
                )
                self.assertTrue(transport.manager.is_pause)
                self.assertTrue(transport.paused)
            finally:
                await transport.close()

        for worker in (False, True):
            with self.subTest(worker=worker):
                asyncio.run(check(worker))

    def test_non_abort_pause_keeps_reader_and_does_not_abort(self):
        async def check(worker):
            transport = _ControlTransport(worker)
            try:
                await transport.start_request()
                await transport.manager.pause_generation(
                    PauseGenerationReqInput(mode="in_place")
                )
                self.assertTrue(await transport.manager.model_update_lock.is_locked())
                self.assertTrue(transport.running)
                self.assertFalse(
                    any(isinstance(obj, AbortReq) for obj in transport.messages)
                )
            finally:
                await transport.close()

        for worker in (False, True):
            with self.subTest(worker=worker):
                asyncio.run(check(worker))

    def test_pd_keeps_existing_path_without_retracting(self):
        async def check(worker, mode):
            transport = _ControlTransport(worker, mode)
            pause_task = None
            try:
                await transport.start_request()
                await transport.manager.pause_generation(
                    PauseGenerationReqInput(mode="in_place")
                )
                count = len(transport.messages)
                pause_task = asyncio.create_task(
                    transport.manager.pause_generation(
                        PauseGenerationReqInput(mode="abort")
                    )
                )
                done, _ = await asyncio.wait({pause_task}, timeout=0.03)
                self.assertFalse(done, "PD drain remains outside this fix")
                self.assertFalse(
                    any(
                        isinstance(obj, PauseGenerationReqInput)
                        and obj.mode == "retract"
                        for obj in transport.messages[count:]
                    )
                )
                self.assertTrue(transport.manager.is_pause)
                self.assertTrue(transport.paused)
                # Explicit external completion, not a claim of PD cleanup.
                transport.finished.set()
                await asyncio.wait_for(pause_task, timeout=2)
            finally:
                if pause_task is not None and not pause_task.done():
                    pause_task.cancel()
                    await asyncio.gather(pause_task, return_exceptions=True)
                await transport.close()

        for worker in (False, True):
            for mode in (DisaggregationMode.PREFILL, DisaggregationMode.DECODE):
                with self.subTest(worker=worker, mode=mode):
                    asyncio.run(check(worker, mode))

    def test_worker_preserves_pause_request_and_single_broadcast(self):
        async def check():
            transport = _ControlTransport(True)
            try:
                await transport.manager.pause_generation(
                    PauseGenerationReqInput(mode="in_place")
                )
                message_count = len(transport.messages)
                broadcast_count = len(transport.callbacks)
                request = PauseGenerationReqInput(
                    mode="abort", rid="control-rid", http_worker_ipc="worker-ipc"
                )
                await transport.manager.pause_generation(request)
                self.assertEqual(request.mode, "abort")
                self.assertEqual(request.rid, "control-rid")
                self.assertEqual(request.http_worker_ipc, "worker-ipc")
                controls = [
                    obj
                    for obj in transport.messages[message_count:]
                    if isinstance(obj, PauseGenerationReqInput)
                ]
                self.assertEqual(len(controls), 1)
                self.assertEqual(controls[0].rid, "control-rid")
                self.assertEqual(controls[0].http_worker_ipc, "worker-ipc")
                self.assertEqual(len(transport.callbacks) - broadcast_count, 1)
                self.assertIsNone(transport.manager._pause_continue_future)
                self.assertTrue(transport.manager.is_pause)
                self.assertTrue(transport.paused)
            finally:
                await transport.close()

        asyncio.run(check())


if __name__ == "__main__":
    unittest.main()
