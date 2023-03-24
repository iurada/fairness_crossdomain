import copy
from collections import deque
import numpy as np
from experiments.classification.SWAD import swa_utils


class SWADBase:
    def update_and_evaluate(self, segment_swa, val_acc, val_loss):
        raise NotImplementedError()

    def get_final_model(self):
        raise NotImplementedError()


class LossValley(SWADBase):
    """IIDMax has a potential problem that bias to validation dataset.
    LossValley choose SWAD range by detecting loss valley.
    """

    def __init__(self, n_converge, n_tolerance, tolerance_ratio, start_it, **kwargs):
        """
        Args:
            evaluator
            n_converge: converge detector window size.
            n_tolerance: loss min smoothing window size
            tolerance_ratio: decision ratio for dead loss valley
        """

        # Ns -> optimum patience
        self.n_converge = n_converge

        # Ne -> overfit patience
        self.n_tolerance = n_tolerance

        # r -> tolerance rate
        self.tolerance_ratio = tolerance_ratio

        # we can force convergence at least from a certain iteration
        self.start_it = start_it

        # set of models, added one at each validation phase
        # length = optimum patience
        self.converge_Q = deque(maxlen=n_converge)

        # set of models, added one at each validation phase
        # length = overfit patience
        self.smooth_Q = deque(maxlen=n_tolerance)

        self.final_model = None

        self.converge_step = None
        self.dead_valley = False
        self.threshold = None

    def get_smooth_loss(self, idx):
        smooth_loss = min([model.end_loss for model in list(self.smooth_Q)[idx:]])
        return smooth_loss

    @property
    def is_converged(self):
        return self.converge_step is not None

    # segment_swa -> the model that contains the models til the current iteration
    def update_and_evaluate(self, segment_swa, val_acc, val_loss):

        if self.dead_valley:
            return

        # copy of the current model
        frozen = copy.deepcopy(segment_swa)

        frozen.end_loss = val_loss

        # fill the queues
        # converge_Q is the queue whose maximum size is Ns
        self.converge_Q.append(frozen)
        #print('converge Q ')
        #print([model.end_loss for model in self.converge_Q])
        # smooth_Q is the queue whose maximum size is Ne
        self.smooth_Q.append(frozen)
        #print('smooth Q ')
        #print([model.end_loss for model in self.smooth_Q])

        # is the converge_step is None (so self.converge_Q is still growing)
        if not self.is_converged:

            # until the converge_Q queue is not full I cannot go on
            if len(self.converge_Q) < self.n_converge:
                return

            # among the models in the queue I search of the index that correspond to the model with the lowest validation loss
            min_idx = np.argmin([model.end_loss.cpu() for model in self.converge_Q])
            #print([model.end_loss for model in self.converge_Q])
            # I take the model with the lowest validation loss
            untilmin_segment_swa = self.converge_Q[min_idx]  # until-min segment swa.

            # row 7 in pseudo code B.4
            # when I encounter a sequence of Ns models in which the first is the lowest I can say that the model has converged
            # so I can assign ts and l
            if min_idx == 0:
                if self.converge_Q[0].end_step < self.start_it: return #! [Forcing convergence at least @ step=start_it]
                # it is used only to stop the self.converge_Q updating since a good window of size Ns is found
                self.converge_step = self.converge_Q[0].end_step
                # start averaging models from the model in ts -> self.converge_Q[min_idx]
                self.final_model = swa_utils.AveragedModel(untilmin_segment_swa)

                # row 9 in pseudo code B.4
                # the mean of all the validation losses from ts to ts+Ns
                th_base = np.mean([model.end_loss.item()  for model in self.converge_Q])
                # th_base * (self.tolerance_ratio)
                self.threshold =  (self.tolerance_ratio) * th_base

                # if Ne is lower then Ns I can exploit the already collected models in self.converge_Q
                if self.n_tolerance < self.n_converge:
                    for i in range(self.n_converge - self.n_tolerance):
                        model = self.converge_Q[1 + i]
                        # I do the average between the models from self.converge_Q[1] to self.converge_Q[Ne]
                        self.final_model.update_parameters(model, start_step=model.start_step, end_step=model.end_step)

                # otherwise
                elif self.n_tolerance > self.n_converge:
                    converge_idx = self.n_tolerance - self.n_converge

                    # take all the elements of the list
                    Q = list(self.smooth_Q)[: converge_idx + 1]
                    #print('Q ')
                    #print([model.end_loss for model in list(self.smooth_Q)[: converge_idx + 1]])
                    start_idx = 0
                    for i in reversed(range(len(Q))): # i goes from the index of the last element in Q to 0
                        model = Q[i]
                        # row 10 in pseudo code B.4
                        # if the validation loss of the last model in smooth Q is higher then the tolerance then we can stop
                        if model.end_loss > self.threshold:
                            # row 11-12 in pseudo code B.4 te (end it. for averaging) assegnation
                            start_idx = i + 1
                            break

                    #print('Start computing the average of the models from ', model.start_step)
                    for model in Q[start_idx + 1 :]:
                        # row 13 pseudo code B.4
                        self.final_model.update_parameters(model.module.model, start_step=model.start_step, end_step=model.end_step)
                        #print(model.end_loss)
                print(
                    f"Model converged at step {self.converge_step}, "
                    f"Start step = {self.final_model.start_step}; "
                    f"Threshold = {self.threshold:.6f}, "
                )
            return

        # self.converge_step is the iteration from which we can start counting Ne -> windows to select te
        #print('self.smooth_Q[0].end_step ',self.smooth_Q[0].end_step)
        if self.smooth_Q[0].end_step < self.converge_step:
            return

        # converged -> loss valley
        # compute the val loss of the first model in self.smooth_Q
        min_vloss = self.get_smooth_loss(0)
        # if it is higher than the threshold (line 9 of pseudo code)
        if min_vloss > self.threshold:
            self.dead_valley = True
            print(f"Valley is dead at step {self.final_model.end_step}")
            return

        model = self.smooth_Q[0]
        self.final_model.update_parameters(
            model.module.model, start_step=model.start_step, end_step=model.end_step
        )

    def get_final_model(self):
        if not self.is_converged:
            print("Requested final model, but model is not yet converged; return last model instead")
            return self.converge_Q[-1]

        if not self.dead_valley:
            self.smooth_Q.popleft()
            while self.smooth_Q:
                smooth_loss = self.get_smooth_loss(0)
                if smooth_loss > self.threshold:
                    break
                segment_swa = self.smooth_Q.popleft()
                self.final_model.update_parameters(segment_swa.module.model, step=segment_swa.end_step)

        return self.final_model