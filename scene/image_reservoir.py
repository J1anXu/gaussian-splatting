import threading
import queue

class ImageReservoir:
    def __init__(self, cameras, capacity=8, device="cuda"):
        self.cameras = cameras
        self.queue = queue.Queue(maxsize=capacity)
        self.device = device
        self._stop = False

        self.thread = threading.Thread(
            target=self._producer_loop,
            daemon=True
        )
        self.thread.start()
        
    def _producer_loop(self):
        import random
        while not self._stop:
            if self.queue.full():
                continue

            cam = random.choice(self.cameras)

            # CPU 侧 IO + resize
            img = cam.load_image_cpu()  # 建议你拆出来
            img = img.to(self.device, non_blocking=True)

            self.queue.put((cam, img))
