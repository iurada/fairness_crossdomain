# Burrowed from https://github.com/pytorch/pytorch/blob/master/torch/optim/swa_utils.py
# modified for the DomainBed.
import copy
import torch
from torch.nn import Module
from copy import deepcopy


class AveragedModel(Module):
    def __init__(self, model, device=None, avg_fn=None):
        super(AveragedModel, self).__init__()

        self.start_step = -1
        self.end_step = -1

        if isinstance(model, AveragedModel):
            model = model.module

        self.module = deepcopy(model)

        # updated at each iteration
        self.register_buffer("n_averaged", torch.tensor(0, dtype=torch.long, device=device))

        # the average: current_avg_weight + (current_weight - current_avg_weight)/num_models
        if avg_fn is None:
            def avg_fn(averaged_model_parameter, model_parameter, num_averaged):
                return averaged_model_parameter + (model_parameter - averaged_model_parameter) / (num_averaged + 1)

        self.avg_fn = avg_fn


    # update parameters is done at each iteration
    # step -> current iteration
    def update_parameters(self, model, step=None, start_step=None, end_step=None):
        """Update averaged model parameters
        Args:
            model: current model to update params
            step: current step. step is saved for log the averaged range
            start_step: set start_step only for first update
            end_step: set end_step
        """

        #print('current it ',step)

        if isinstance(model, AveragedModel):
            model = model.module

        # for each parameter in the model
        for p_swa, p_model in zip(self.parameters(), model.parameters()):

            device = p_swa.device
            p_model_ = p_model.detach().to(device)
            # if it is the first iteration after the initialization of AveragedModel, so the number of averaged model is 0
            if self.n_averaged == 0:
                # I copy the model as it is in p_swa (which is where I collect the average model)
                p_swa.detach().copy_(p_model_)
            else:
                # if it's from the second iteration on, I copy in p_swa the mean between the current model and the previous ones
                p_swa.detach().copy_(self.avg_fn(p_swa.detach(), p_model_, self.n_averaged.to(device)))

        # I increase the number of the averaged model
        self.n_averaged += 1

        if step is not None:
            if start_step is None:
                # I assign the start step to the current iteration if it has not been assigned before (if it's the first iteration
                # after the initialization of AveragedModel)
                start_step = step
            if end_step is None:
                # I assign the end step to the current iteration at every iteration
                end_step = step

        if start_step is not None:
            if self.n_averaged == 1:
                self.start_step = start_step

        if end_step is not None:
            self.end_step = end_step

        #print('self.start_step ',self.start_step)
        #print('self.n_averaged', self.n_averaged)


@torch.no_grad()
def update_bn(iterator, model, n_steps, device="cuda"):
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None
        module.num_batches_tracked *= 0

    for i in range(n_steps):
        # batches_dictlist: [{env0_data_key: tensor, env0_...}, env1_..., ...]
        batches_dictlist = next(iterator)
        x = torch.cat([dic["x"] for dic in batches_dictlist])
        x = x.to(device)

        model(x)

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)