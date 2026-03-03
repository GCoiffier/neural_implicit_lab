---
title: Callbacks
weight: 5
---

Callback are small piece of logic that affect the trainer they are associated with, provide log infos, or do anything you can think of.  
Inside a Trainer, they can be called at four points:   

- At the beginning of an training epoch  
- At the end of an training epoch  
- At the end of a forward/backward pass  
- At the end of a testing epoch  

Callbacks are added at the beginning of the training using the `add_callbacks` method of the `Trainer` class:

```python
from implicitlab.training import SimpleRegressionTrainer, TrainingConfig
from implicitlab.training import callbacks

trainer = SimpleRegressionTrainer(TrainingConfig(), lossfun=torch.nn.MSELoss())
trainer.add_callbacks(
    callbacks.LoggerCB("output/training_log.txt"), # write losses and info about training in a .txt file
    callbacks.Render2DCB("output", 10) # makes a snapshot of a 2D neural implicit every 10 training epochs
)
```

## List of implemented callbacks

:::implicitlab.training.callbacks
    options:
        heading_level: 3
        filters:
        - "!Callback"


## Write your own callback

All callbacks inherit from the base class `Callback`, which implements four methods:

```python
class Callback:
    def callOnBeginTrain(self, trainer, model): pass
    
    def callOnEndTrain(self, trainer, model): pass
    
    def callOnEndForward(self, trainer, model): pass
    
    def callOnEndTest(self, trainer, model): pass
```

Every method takes as argument the trainer it's linked to, and the model the trainer is currently optimizing. Therefore, it has full information about what's going on during optimization.

To implement our own custom callback, simply create a class that inherits from `Callback` and that implements one or several of these methods.
