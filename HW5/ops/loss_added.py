import torch
import torch.nn as nn
from torch.autograd import Variable


class GANLoss(nn.Module):
    def __init__(
        self,
        use_lsgan=True,
        target_real_label=1.0,
        target_fake_label=0.0,
        tensor=torch.FloatTensor,
    ):
        super().__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor

        self.loss = nn.MSELoss() if use_lsgan else nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None

        create_f = lambda var_t: (var_t is None) or (var_t.numel() != input.numel())

        if target_is_real:
            create_label = create_f(self.real_label_var)

            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)

            return self.real_label_var
        # else
        create_label = create_f(self.fake_label_var)

        if create_label:
            fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
            self.fake_label_var = Variable(fake_tensor, requires_grad=False)
        return self.fake_label_var

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)
