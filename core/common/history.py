import numpy as np


class HistoryHandler:
    def __init__(self, history_config, obs_dims):
        self.obs_dims = obs_dims
        self.history = {}

        self.buffer_config = {}
        for obs_key, obs_num in history_config.items():
            if obs_key in self.buffer_config:
                self.buffer_config[obs_key] = max(self.buffer_config[obs_key], obs_num)
            else:
                self.buffer_config[obs_key] = obs_num

        print("\nHistoryHandler: initializing buffers")
        for key, T in self.buffer_config.items():
            dim = self.obs_dims[key]
            print(f"  key={key}: T={T}, dim={dim}")
            self.history[key] = np.zeros((1, T, dim), dtype=np.float32)

        print("HistoryHandler: done\n")

    def reset(self, reset_ids):
        if len(reset_ids) == 0:
            return
        assert set(self.buffer_config.keys()) == set(self.history.keys())
        for key in self.history.keys():
            self.history[key][reset_ids] *= 0.0

    def add(self, key: str, value: np.ndarray):
        assert key in self.history.keys(), f"Key {key} not found in history"
        val = self.history[key][:]
        self.history[key][:, 1:] = val[:, :-1]
        self.history[key][:, 0] = value[:]

    def query(self, key: str):
        assert key in self.history.keys(), f"Key {key} not found in history"
        return self.history[key][:]
