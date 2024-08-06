import torch
from axonn.intra_layer import overlap_all_gathers_for_checkpointed_forward
from torch.utils.checkpoint import detach_variable
from torch.cuda.amp import custom_fwd, custom_bwd

class CheckpointFunction(torch.autograd.Function):
    """This function is adapted from torch.utils.checkpoint with
       two main changes:
           1) torch.cuda.set_rng_state is replaced with `_set_cuda_rng_state`
           2) the states in the model parallel tracker are also properly
              tracked/set/reset.
    """

    @staticmethod
    @custom_fwd
    def forward(ctx, run_function, checkpointed_model, *args):
        ctx.run_function = run_function
        ctx.checkpointed_model = checkpointed_model

        # Copy the rng states.
        ctx.fwd_cpu_rng_state = torch.get_rng_state()
        ctx.fwd_cuda_rng_state = torch.cuda.get_rng_state()

        with torch.no_grad():
            outputs = run_function(*args)

        ctx.save_for_backward(*args)

        return outputs

    @staticmethod
    @custom_bwd
    def backward(ctx, *args):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad(), "
                "please use .backward() if possible"
            )
        inputs = ctx.saved_tensors

        # Store the current states.
        bwd_cpu_rng_state = torch.get_rng_state()
        bwd_cuda_rng_state = torch.cuda.get_rng_state()

        # Set the states to what it used to be before the forward pass.
        torch.set_rng_state(ctx.fwd_cpu_rng_state)
        torch.cuda.set_rng_state(ctx.fwd_cuda_rng_state)

        # Compute the forward pass.
        detached_inputs = detach_variable(inputs)
        with torch.enable_grad():
            with overlap_all_gathers_for_checkpointed_forward(ctx.checkpointed_model):
                outputs = ctx.run_function(*detached_inputs)

        # Set the states back to what it was at the start of this function.
        torch.set_rng_state(bwd_cpu_rng_state)
        torch.cuda.set_rng_state(bwd_cuda_rng_state)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        torch.autograd.backward(outputs, args)
        grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else inp for inp in detached_inputs)
        return (None, None) + grads


def checkpoint(function, checkpointed_model, *args):
    """Checkpoint a model or part of the model.
    This has been directly copied from torch.utils.checkpoint."""
    return CheckpointFunction.apply(function, checkpointed_model, *args)
