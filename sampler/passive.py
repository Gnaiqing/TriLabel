from base import BaseSampler


class PassiveSampler(BaseSampler):
    def sample_distinct(self, n=1):
        N = len(self.train_data)
        n_sampled = 0
        indices = []
        while n_sampled < n:
            idx = self.rng.choice(N)
            if self.sampled[idx]:
                continue
            self.sampled[idx] = True
            indices.append(idx)
            n_sampled += 1

        labels = self.label_selected_indices(indices)
        return indices, labels

