import concurrent.futures
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class StopWorker(BaseException):
    pass


class TestMooncakeTransferDevice(unittest.TestCase):
    def manager(self, enabled=True, pool="INTRA_NODE_NVLINK", gpu_id=3):
        manager = object.__new__(MooncakeKVManager)
        manager.enable_custom_mem_pool = enabled
        manager.custom_mem_pool_type = pool
        manager.kv_args = SimpleNamespace(gpu_id=gpu_id)
        manager.enable_trace = False
        return manager

    def test_worker_binds_local_device_before_dequeue_on_worker_thread(self):
        manager = self.manager()
        calls = []
        main_thread = threading.get_ident()

        class Queue:
            def get(self):
                calls.append(("dequeue", threading.get_ident()))
                raise StopWorker

        def run():
            try:
                manager.transfer_worker(Queue(), None)
            except StopWorker:
                pass

        with patch(
            "sglang.srt.disaggregation.mooncake.conn.torch.cuda.set_device",
            side_effect=lambda device: calls.append(
                ("bind", threading.get_ident(), device)
            ),
        ):
            thread = threading.Thread(target=run)
            thread.start()
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], ("bind", calls[1][1], 3))
        self.assertEqual(calls[1][0], "dequeue")
        self.assertNotEqual(calls[0][1], main_thread)

    def test_executor_initializer_binds_each_thread_before_transfer(self):
        manager = self.manager(gpu_id=2)
        local = threading.local()
        barrier = threading.Barrier(2)

        def transfer():
            barrier.wait(timeout=5)
            return threading.get_ident(), local.device

        with patch(
            "sglang.srt.disaggregation.mooncake.conn.torch.cuda.set_device",
            side_effect=lambda device: setattr(local, "device", device),
        ):
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=2, initializer=manager.init_transfer_thread_device
            ) as executor:
                futures = [executor.submit(transfer) for _ in range(2)]
                results = [f.result(timeout=5) for f in futures]
        self.assertEqual({device for _, device in results}, {2})
        self.assertEqual(len({tid for tid, _ in results}), 2)

    def test_non_intra_node_transports_do_not_initialize_cuda(self):
        for enabled, pool in [(False, None), (False, "INTRA_NODE_NVLINK"),
                              (True, "NVLINK"), (True, "BAREX")]:
            with self.subTest(enabled=enabled, pool=pool):
                with patch(
                    "sglang.srt.disaggregation.mooncake.conn.torch.cuda.set_device"
                ) as set_device:
                    self.manager(enabled, pool).init_transfer_thread_device()
                    set_device.assert_not_called()


if __name__ == "__main__":
    unittest.main()
