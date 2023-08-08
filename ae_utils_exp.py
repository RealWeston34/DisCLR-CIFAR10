import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

def one_cold(dim, ind):
    out = torch.ones(dim)
    out[ind] = 0
    return out

def covariance(X, Y): # where each is shape (N, L)
    cov = (X*Y).mean(dim=0) - X.mean(dim=0)*Y.mean(dim=0)
    return cov

def s_init(module):
    torch.nn.init.kaiming_normal_(module.weight, nonlinearity='linear')
    return module

class Normalize(object):
    def __init__(self, mean, std, ndim=2):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
        for d in range(ndim):
            self.mean = self.mean.unsqueeze(-1)
            self.std = self.std.unsqueeze(-1)

    def __call__(self, tensor):
        return tensor.sub(self.mean.to(tensor.device)).div(self.std.to(tensor.device))
    
class UnNormalize(object):
    def __init__(self, norm):
        super(UnNormalize, self).__init__()
        self.mean = norm.mean
        self.std = norm.std

    def __call__(self, tensor):
        return self.scale_inorm(tensor).add(self.mean.to(tensor.device))

    def scale_inorm(self, tensor):
        return tensor.mul(self.std.to(tensor.device))

cifar10_norm = Normalize(mean = 0.4734, std= 0.2516)
cifar10_inorm = UnNormalize(cifar10_norm)


class InvNorm(object):
    
    def __init__(self, norm):
        super(InvNorm, self).__init__()
        self.mean = 0.
        self.std = 1.
        if hasattr(norm, 'running_mean'):
            self.mean = norm.running_mean
        if hasattr(norm, 'running_var'):
            self.std = norm.running_var.sqrt()
        
    def __call__(self, x):
        # expect batch, ...
        dim = x.dim()
        mean = self.mean.clone()
        std = self.std.clone()
        for i in range(dim - 2):
            mean.unsqueeze_(-1)
            std.unsqueeze_(-1)
        return (x + mean) * std

class Predictor(torch.nn.Module):
    
    def __init__(self, config=(10, 20, 20)):
        super(Predictor, self).__init__()
        self.ops = []
        for i in range(len(config)-1):
            self.ops.append(s_init(torch.nn.Linear(config[i], config[i+1], bias=True)))
            self.ops.append(torch.nn.SELU(inplace=True))
        self.ops.append(s_init(torch.nn.Linear(config[-1], 1, bias=True)))
        self.op = torch.nn.Sequential(*self.ops)
        
    def forward(self, x):
        return self.op(x)
    
class PredictorEnsemble(torch.nn.Module):
    
    def __init__(self, n_preds=2, p_h_config=(40, 40)):
        super(PredictorEnsemble, self).__init__()
        self.n_preds = n_preds
        self.preds = torch.nn.ModuleList([Predictor(config=(n_preds,) + p_h_config) for i in range(n_preds)])
        
    def forward(self, latent_var):
        predictions = torch.empty_like(latent_var)
        for i in range(self.n_preds): # mask the ground truth
            mask = one_cold(latent_var.shape, (..., i)).to(latent_var.device)
            preds_out = self.preds[i](latent_var * mask).squeeze()
            predictions[..., i] = preds_out
        return predictions
    

class Disentangler(torch.nn.Module):
    
    def __init__(self, device, inp_norm, inp_inorm=None, feature_dim=512, z_dim=2, p_h_config=(40, 40), z_act=torch.nn.Sigmoid()):
        super(Disentangler, self).__init__()
        self.z_dim = z_dim
        self.enc = nn.Sequential(nn.Linear(feature_dim, 2048),
                                 nn.ReLU(),
                                 nn.Linear(2048, z_dim))
        
        self.dec = nn.Sequential(nn.Linear(z_dim, 2048),
                                 nn.ReLU(),
                                 nn.Linear(2048, feature_dim))
        self.z_act = z_act
        # self.inp_inorm = inp_inorm
        self.preds = PredictorEnsemble(n_preds=self.z_dim, p_h_config=p_h_config)
        self.ae_optim=None
        self.preds_optim=None
        self.device=device
        self.inp_norm = inp_norm
        self.inp_inorm = inp_inorm
        self.to(device)
        
    def create_encoding(self, x):
        self.x_norm = self.inp_norm(x)
        return self.z_act(self.enc(self.x_norm))
        
    def forward(self, x):
        self.z = self.create_encoding(x)
        self.z_pred = self.preds(self.z)
        self.x_pred = self.dec(self.z)
        return self.x_pred

    def predict_z(self, x):
        z = self.create_encoding(x)
        z_pred = self.preds(z)
        return z_pred

    
    def compute_loss_ae(self, expected):
        # compute reconstruction loss
        rec_loss = 0.5*self.mse(self.x_pred, expected.detach())
        # compute covariance score
        adv_loss = covariance(self.z.detach(), self.z_pred).sum()
        
        return rec_loss, adv_loss
    
    def compute_loss_preds(self):
        return 0.5*self.mse(self.z_pred, self.z.detach())
    
    def step_ae(self, expected, ar=0.0):
        self.enc.zero_grad()
        self.dec.zero_grad()
        rec_loss, adv_loss = self.compute_loss_ae(expected)
        loss = (1. - ar)*(rec_loss) + ar*adv_loss
        loss.backward()
        self.ae_optim.step()
        return rec_loss, adv_loss
    
    def step_preds(self):
        self.preds.zero_grad()
        preds_loss = self.compute_loss_preds()
        preds_loss.backward()
        self.preds_optim.step()
        return preds_loss
    
    def init_optim_objects(self, lr, pred_lr):
        self.ae_optim = torch.optim.Adam(list(self.enc.parameters()) + list(self.dec.parameters()), lr=lr)
        self.preds_optim = torch.optim.Adam(self.preds.parameters(), lr=pred_lr)
    
    def fit(self, dataset, n_group, batch_per_group=10, lr=0.001, pred_lr=0.01, ar=0.0,
            batch_size=100, preds_train_iters=5, generator_ae=None):
        
        self.train()
        # create loss objects and optimizers
        self.mse = torch.nn.MSELoss()
        if self.ae_optim is None or self.preds_optim is None:
            self.init_optim_objects(lr, pred_lr)
        # set up loss storage
        rec_loss = torch.zeros(n_group)
        adv_loss = torch.zeros(n_group)
        pred_loss = torch.zeros(n_group)
        # define samplers for the AE
        n_samples = batch_size*batch_per_group*n_group
        random_sampler_ae = torch.utils.data.RandomSampler(dataset, \
                      replacement=True, num_samples=n_samples, generator=generator_ae)
        batch_sampler_ae = torch.utils.data.BatchSampler(random_sampler_ae, batch_size=batch_size, drop_last=True)
        dataloader_ae = iter(torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler_ae))
        
        for g in range(n_group):
            rec_loss_agg = 0.
            adv_loss_agg = 0.
            pred_loss_agg = 0.
            for b in range(batch_per_group):
                # print("\rGroup: {}\t{:2.0f}%".format(g, 100*(b+1)/batch_per_group), end="")
                data, y = next(dataloader_ae)
                # push examples through the Disentangler, get latent space activations
                ex = data.to(self.device)
                ex_no_grad = ex.detach()
                out = self.forward(ex_no_grad)
                for p in range(preds_train_iters):
                    self.z_pred = self.preds(self.z.detach())
                    pred_loss_agg += self.step_preds() / preds_train_iters
                out = self.forward(ex_no_grad)
                rec_loss_b, adv_loss_b = self.step_ae(self.x_norm.detach(), ar)
                rec_loss_agg += rec_loss_b
                adv_loss_agg += adv_loss_b
            rec_loss[g] = rec_loss_agg / batch_per_group
            adv_loss[g] = adv_loss_agg / batch_per_group
            pred_loss[g] = pred_loss_agg / batch_per_group
            # print("\tRec: {:1.4f}\tAdv: {:1.4f}\tPred: {:1.4f}".format(\
            #     rec_loss[g], adv_loss[g], pred_loss[g]))
        return rec_loss.detach(), adv_loss.detach(), pred_loss.detach()
    
    def n_fit(self, dataset, n_group, batch_per_group=10, lr=0.001, pred_lr=0.01, ar=0.0,
                batch_size=100, preds_train_iters=5, generator_ae=None):
            
            self.train()
            # create loss objects and optimizers
            self.mse = torch.nn.MSELoss()
            if self.ae_optim is None or self.preds_optim is None:
                self.init_optim_objects(lr, pred_lr)
            # set up loss storage
            rec_loss = torch.zeros(n_group)
            adv_loss = torch.zeros(n_group)
            pred_loss = torch.zeros(n_group)
            # define samplers for the AE
            n_samples = batch_size*batch_per_group*n_group
            random_sampler_ae = torch.utils.data.RandomSampler(dataset, \
                        replacement=True, num_samples=n_samples, generator=generator_ae)
            batch_sampler_ae = torch.utils.data.BatchSampler(random_sampler_ae, batch_size=batch_size, drop_last=False)
            dataloader_ae = iter(torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler_ae))
            
            for g in range(n_group):
                rec_loss_agg = 0.
                adv_loss_agg = 0.
                pred_loss_agg = 0.
                for b in range(batch_per_group):
                    print("\rGroup: {}\t{:2.0f}%".format(g, 100*(b+1)/batch_per_group), end="")
                    data, y = next(dataloader_ae)
                    # push examples through the Disentangler, get latent space activations
                    ex = data.to(self.device)
                    ex_no_grad = ex.detach()
                    # ----------Input into backbone encoder----------
                    sizes = ex_no_grad.size()
                    ex_no_grad = ex_no_grad.view(sizes[0] * 2, sizes[2], sizes[3], sizes[4])


                    # load pre-trained resnet18 model
                    enc = resnet18(weights='DEFAULT')

                    # remove final fully connected layer
                    enc = torch.nn.Sequential(*list(enc.children())[:-1])

                    # move resnet18_encoder to same device as input tensor
                    enc = enc.to(self.device)
                    ex_no_grad = enc(ex_no_grad)
                    ex_no_grad = torch.squeeze(ex_no_grad)
                    
                    out = self.forward(ex_no_grad)
                    for p in range(preds_train_iters):
                        self.z_pred = self.preds(self.z.detach())
                        pred_loss_agg += self.step_preds() / preds_train_iters
                    out = self.forward(ex_no_grad)
                    rec_loss_b, adv_loss_b = self.step_ae(self.x_norm.detach(), ar)
                    rec_loss_agg += rec_loss_b
                    adv_loss_agg += adv_loss_b
                rec_loss[g] = rec_loss_agg / batch_per_group
                adv_loss[g] = adv_loss_agg / batch_per_group
                pred_loss[g] = pred_loss_agg / batch_per_group
                print("\tRec: {:1.4f}\tAdv: {:1.4f}\tPred: {:1.4f}".format(\
                    rec_loss[g], adv_loss[g], pred_loss[g]))
            return rec_loss.detach(), adv_loss.detach(), pred_loss.detach()
        