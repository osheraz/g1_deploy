import time

class TicToc:
    def __init__(self):
        self.times = {}

    def tic(self, name: str):
        self.times[name] = time.time()

    def toc(self, name: str):
        if name not in self.times:
            print(f"[TicToc] '{name}' was never started.")
            return
        dt = time.time() - self.times[name]
        print(f"[TicToc] {name}: {dt*1000:.3f} ms")
        del self.times[name]