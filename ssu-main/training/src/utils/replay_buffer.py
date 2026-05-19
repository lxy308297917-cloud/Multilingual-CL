import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.n_seen = 0

    def add(self, examples):
        for ex in examples:
            self.n_seen += 1
            if len(self.buffer) < self.capacity:
                self.buffer.append(ex)
            else:
                j = random.randint(0, self.n_seen - 1)
                if j < self.capacity:
                    self.buffer[j] = ex

    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return []
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)
