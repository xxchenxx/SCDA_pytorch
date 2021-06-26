import torch
import numpy as np

def get_ntk_n(model, data_loader, device, num_batch=-1):
    
    model.eval()

    grads = []
    for i, (images, target) in enumerate(data_loader):
        if num_batch > 0 and i >= num_batch: break
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        logit = model(images)

        for _idx in range(len(images)):
            logit[_idx:_idx+1].backward(torch.ones_like(logit[_idx:_idx+1]), retain_graph=True)
            grad = []
            for name, p in model.named_parameters():
                if p.grad is not None:
                    grad.append(p.grad.view(-1).detach())

            grads.append(torch.cat(grad, -1))
            model.zero_grad()
            torch.cuda.empty_cache()

    grads = torch.stack(grads, 0)
    conds_dict = {}    
    conds = []
    n_cur = 0
    for name, p in model.named_parameters():
        if p.grad is not None:
            grad = grads[:, n_cur:n_cur + p.grad.nelement()]
            n_cur += p.grad.nelement()
            ntk = torch.einsum('nc,mc->nm', [grad, grad])
            eigenvalues, _ = torch.symeig(ntk) 
            condition_number = np.nan_to_num((eigenvalues[-1] / eigenvalues[0]).item(), copy=True, nan=100000.0)
            conds.append(condition_number)
            conds_dict[name] = condition_number
    print('Mead condition-number-ntk = {}'.format(np.mean(np.array(conds))))
    return conds_dict

