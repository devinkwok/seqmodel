import time
import torch


class Action():

    """
    Generic container for functions to run periodically in test/train loop.
    Arguments from calling object are passed on to child actions.

    Args:
        child_actions: list of actions to run with this one
        iter_period: how many iterations between runs, set to `0` to ignore
        time_period: how long to wait between runs (wall time), set to `0` to ignore
        run_at_start: if `True`, always run on first batch
        run_at_end: if `True`, always run on last batch
    """
    def __init__(self, child_actions, iter_period=0, time_period=0.,
                        run_at_start=True, run_at_end=True):
        self.child_actions = child_actions
        self.iter_period = iter_period
        self.time_period = time_period
        self.run_at_start = run_at_start
        self.run_at_end = run_at_end
        self.last_runtime = None

    # must provide n_iter and cur_time
    def __call__(self, **kwargs):
        self.run(**kwargs)

    # required: n_iter, cur_time
    # overrideable: is_Start, is_end
    def run(self, n_iter=None, cur_time=None, is_start=False, is_end=False, **kwargs):
        early_stop = False
        if is_start:
            self.set_up(**kwargs)
        if (is_start and self.run_at_start) or \
            (is_end and self.run_at_end) or \
            (self.iter_period > 0 and n_iter > 0 and n_iter % self.iter_period == 0) or \
            (self.time_period > 0. and cur_time - self.last_runtime > self.time_period):

            self.last_runtime = cur_time
            for action in self.child_actions:
                early_stop += action(n_iter=n_iter, cur_time=cur_time,
                                is_start=is_start, is_end=is_end, **kwargs)
        if is_end:
            self.tear_down(n_iter=n_iter, cur_time=cur_time, **kwargs)
        return early_stop

    def set_up(self, **kwargs):
        self.last_runtime = cur_time  # set runtime at start
        for action in self.child_actions:
            action.set_up(**kwargs)

    # required: n_iter, cur_time
    def tear_down(self, n_iter=None, cur_time=None, **kwargs):
        self.last_runtime = -1.
        for action in self.child_actions:
            action.tear_down(n_iter=n_iter, cur_time=cur_time, **kwargs)


class EpochLoop(Action):

    # put BatchLoop in child_actions
    def __init__(self, child_actions, n_iterations=1, iter_offset=0):
        super().__init__(child_actions, iter_period=1)
        self.n_iterations = n_iterations
        self.iter_offset = iter_offset

    # required: n_iter
    # replaced: cur_time, is_start, is_end
    def run(self, n_iter=None, cur_time=None, is_start=None, is_end=None, **kwargs):
        cur_time = time.time()
        is_start = (n_iter == 0)
        is_end = (n_iter == self.n_iterations - 1)

        early_stop = super().run(n_iter=n_iter + self.iter_offset,
                    cur_time=cur_time, is_start=is_start, is_end=is_end, **kwargs)
        return early_stop

    # overrides Action()
    def __call__(self, **kwargs):
        self.loop(**kwargs)

    # replaced: n_iter
    def loop(self, n_iter=None, **kwargs):
        for epoch in range(self.n_iterations):
            early_stop = self.run(n_iter=epoch, **kwargs)
            if early_stop:
                return early_stop
        return False  # indicates no early stopping


class BatchLoop(EpochLoop):

    """
    Generic training loop which loops through batched data
    and calls a list of Action objects.

    Args:
        task: Task object to optimize
        data_loader: iterable source of batched data
        device: where to send data
        iter_offset: initial value of number of iterations
        child_actions: list of actions to apply with arguments task, batch
                put `Train` in this list
    """
    def __init__(self, data_loader, device, child_actions, iter_offset=0):
        super().__init__(child_actions, n_iterations=len(data_loader), iter_offset=iter_offset)
        self.data_loader = data_loader
        self.device = device

    def loop(self, **kwargs):
        for i, batch in enumerate(self.data_loader):
            batch = batch.to(self.device)
            early_stop = self.run(batch=batch, n_iter=i, **kwargs)
            del batch
            if early_stop:
                return early_stop
        return False  # indicates no early stopping


class EvalLoop(BatchLoop):

    def __init__(self, data_loader, device, child_actions, iter_offset=0):
        super().__init__(data_loader, device, child_actions, iter_offset=iter_offset)

    def loop(self, batch=None, task=None, predicted=None, target=None, latent=None, loss=None,
                n_iter=None, cur_time=None, is_start=False, is_end=False, **kwargs):
        super().loop(task=task, **kwargs)

    def accumulate(self, batch=batch, task=task):
        pass


class Train(Action):

    # put Action containing Evaluator in child_actions
    def __init__(self, task, device, optimizer, child_actions, scheduler=None, n_accumulate=1, n_batch=0):
        super().__init__([self.step] + child_actions + [self.zero_grad],
                        iter_period=n_accumulate)
        self.device = device
        self.task = task.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.n_accumulate = n_accumulate
        self.n_batch = n_batch

    def run(self, batch, n_iter=None, cur_time=None, is_start=False, is_end=False):
        self.task.train()
        predicted, target, latent, loss = self.task.loss(batch)
        loss = loss / self.n_accumulate
        loss.backward()
        cur_time = time.time()
        # run step(), other actions, zero_grad() if enough accumulated gradients
        # and other tasks such as log loss (sum up), early stopping check
        # replace iteration number with batch number (tracked in this object)
        super().run(batch, self.task, predicted, target, latent, loss,
                n_iter=self.n_batch, cur_time=cur_time, is_start=is_start, is_end=is_end)
        del predicted, target, latent, loss

    # update model before other actions
    def step(self, *args, **kwargs):
        self.optimizer.step()
        if not (self.scheduler is None):
            self.scheduler.step()
    
    # zero gradients after other actions
    # this allows gradients to be accessed by other actions
    def zero_grad(self, *args, **kwargs):
        self.optimizer.zero_grad()
        self.n_batch += 1



# create Action with Evaluator in child_actions to schedule evaluations, put this in Train
class Evaluate(Action):

    # put Logger in child_actions
    # these tasks are only performed after evaluation is complete
    def __init__(self, valid_loader, device, child_actions, test_loader=None,
                iter_period=0, time_period=0., run_at_start=False, run_at_end=True):
        self.valid_loop = EvalLoop(valid_loader, device, self.eval_accumulators)
        super(Action, self).__init__([self.valid_loop], iter_period=0, time_period=0.,
                        run_at_start=True, run_at_end=True):
        if test_loader is None:
            self.test_loop = None
        else:
            self.test_loop = EvalLoop(test_loader, device, accumulators)

    def run(batch, task, predicted, target, latent, loss,
            n_iter=self.n_batch, cur_time=cur_time, is_start=is_start, is_end=is_end):
        task.eval()

    # in BatchLoop, loop() calls run(), but Evaluator is an Action foremost
    # therefore run() calls loop(), allowing reuse of loop() code conditional on run()
    # rewrite run_actions() so it does not call run()
    def eval_accumulators(batch, n_iter=i)
        correct = task.evaluate(batch)

    def loop(self, task, batch):
        self.run(task, batch, correct, n_iter=n_iter + self.iter_offset,
                    cur_time=cur_time, is_start=is_start, is_end=is_end)
        # aggregate stats
        self.n_correct += correct
        self.n_total += len(batch)

    def tear_down(self, *args, n_iter, cur_time):
        super().tear_down(*args, n_iter, cur_time)
        # move to CPU
        #TODO log accuracy, other stats
        # log predictions if relevant
        # log model state as snapshot

        if not (self.test_loop is None):
            self.test_loop()
        pass


class Predictor(BatchLoop):

    def __init__(self, task, data_loader, device):
        super().__init__(task, data_loader, None, device)

    def run(self, batch):
        self.task.eval()
        predicted, latent = self.task.forward(batch)
        #TODO do something with predicted, latent
