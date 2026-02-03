import threading
import queue
import time
import random


class ImageReservoir:
    # def __init__(self, cameras, capacity=8, device="cuda"):
    #     self.cameras = cameras
    #     self.queue = queue.Queue(maxsize=capacity)
    #     self.device = device
    #     self._stop = False
    #     self.thread = threading.Thread(
    #         target=self._producer_loop,
    #         daemon=True
    #     )
    #     self.thread.start()
    #     print("[ImageReservoir] thread started:", self.thread.is_alive())

    def __init__(self, cameras, capacity=8, device="cuda", num_workers=4):
        self.cameras = cameras
        self.queue = queue.Queue(maxsize=capacity)
        self.device = device
        self._stop = False
        self.threads = []

        for i in range(num_workers):
            t = threading.Thread(target=self._producer_loop, daemon=True)
            t.start()
            self.threads.append(t)

        print(f"[ImageReservoir] {len(self.threads)} producer threads started")

    # def _producer_loop(self):
    #     print("[Producer] thread entered")
    #     try:
    #         while not self._stop:
    #             if self.queue.full():
    #                 time.sleep(0.001)
    #                 continue
    #             cam = random.choice(self.cameras)
    #             img_cpu = cam.load_image_cpu()
    #             img_gpu = img_cpu.to(self.device, non_blocking=True)
    #             self.queue.put((cam, img_gpu))
    #     except Exception as e:
    #         print("[Producer] EXCEPTION:", e)
    #         import traceback
    #         traceback.print_exc()
    

    def _producer_loop(self):
        while not self._stop:
            if self.queue.full():
                time.sleep(1)
                continue

            cam = random.choice(self.cameras)

            # CPU only
            # 很可能出现“你还在用，生产者已经把这块 pinned buffer 复用了 / 覆盖了”
            img_cpu = cam.load_image_cpu().clone().pin_memory()

            # ❌ 不要 to(cuda)
            self.queue.put((cam, img_cpu))

