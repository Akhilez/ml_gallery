class LinearDecay:
    def __init__(self):
        self.decrease = None

    def __call__(self, config, log):
        step = log.step
        flatten_step = config.epsilon_flatten_step
        start = config.epsilon_start
        end = config.epsilon_end
        decrease = self.decrease or self._calculate_decrease(start, end, flatten_step)

        if step >= flatten_step:
            return end
        return start + (decrease * step)

    def _calculate_decrease(self, start, end, flatten_step):
        self.decrease = (end - start) / flatten_step
        return self.decrease


LINEAR = "linear"

decay_functions = {
    LINEAR: LinearDecay(),
}
