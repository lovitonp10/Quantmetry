import numpy as np
from gluonts.transform.sampler import InstanceSampler


class ExpectedValidationInstanceSampler(InstanceSampler):
    step: int = 6
    context: int = 164

    def __call__(self, ts: np.ndarray) -> np.ndarray:
        a, b = self._get_bounds(ts)

        window_size = self.context
        list_indices = []
        while window_size < b - self.step - 1:
            list_indices.append(window_size)
            window_size = window_size + self.step

        return np.array(list_indices)


def ValidationSplitSamplerIncremental(
    axis: int = -1,
    min_past: int = 0,
    min_future: int = 0,
    step: int = 6,
    context: int = 164,
) -> ExpectedValidationInstanceSampler:
    return ExpectedValidationInstanceSampler(
        axis=axis, min_past=min_past, min_future=min_future, step=step, context=context
    )
