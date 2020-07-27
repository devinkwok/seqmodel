loader
    - manages hyperparameters and default states for optuna
    - takes configs
    - creates objects
trainer
    - splits data, sends to task
    - data goes to GPU here
    - reports metrics
    - saves checkpoints
task
    - dictates data required
    - preprocesses data
    - runs through model
    - gets loss
    loss
        - plug and play modules
model
    - same as current
seq
    - plug and play datasets, loaders
    - utilities


TODO:
 - augment data as a LambdaModule to plug into dataset, do this before dataloader to save GPU cycles
 - handle different data sizes (x,) (x, target) at the task level