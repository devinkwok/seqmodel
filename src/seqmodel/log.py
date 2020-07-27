


class Logger(Action):

    # logs a summary of the object
    def log_summary(self, name, obj, period, time):
        # move to CPU
        pass

    # saves entire object
    def save(self, name, obj, period, time):
        # move to CPU
        pass


class Checkpoint(Action):

    def __init__(self, path, only_save_best=True):
        self.only_save_best = only_save_best
        self.path = path

    def save_checkpoint(self, task, optimizer, loss, epoch, n_iter, time)
        if only_save_best and self.best_loss < loss:
            return False  # do not save
        # move to CPU?
        torch.save({
                'epoch': epoch,
                'model_state_dict': task.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, self.get_name(n_iter))

    def get_name(self, n_iter):
        return self.path + str(n_iter) + '.seqmod'
