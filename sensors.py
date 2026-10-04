from dataclasses import dataclass
import numpy as np


@dataclass
class Sensor():
    kind: str
    ts: float
    data: np.ndarray
    sigma: float | np.ndarray
    last_ts: float
    flag: bool = False

    def new_meas(self, ts, data):
        self.ts, self.data = ts, data

    def has_new_data(self):
        return self.ts > self.last_ts