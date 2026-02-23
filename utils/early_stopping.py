class EarlyStopping:
    """
    Early stopping based on F1.
    Patience is in units of evaluation rounds.
    """
    def __init__(self, patience: int = 3, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score: float | None = None
        self.should_stop = False

    def step(self, val_f1: float) -> bool:
        if self.best_score is None:
            self.best_score = val_f1
        elif val_f1 > self.best_score + self.min_delta:
            self.best_score = val_f1
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop