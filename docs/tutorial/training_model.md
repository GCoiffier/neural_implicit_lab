---
title: Training a neural implicit
---


## Preparing your neural model


## The Trainer class


```python
config = TrainingConfig(
    BATCH_SIZE=10_000,
    TEST_BATCH_SIZE = 10000,
    N_EPOCHS=500,
    LEARNING_RATE=1e-4,
    DEVICE="cuda",
    OPTIMIZER="adam",
)
```

## Training

`Trainer` class