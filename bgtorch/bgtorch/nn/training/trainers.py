import torch
import torch.nn as nn

import numpy as np

from bgtorch.utils.types import assert_numpy


class LossReporter:
    """
        Simple reporter use for reporting losses and plotting them.
    """

    def __init__(self, *labels):
        self._labels = labels
        self._n_reported = len(labels)
        self._raw = [[] for _ in range(self._n_reported)]

    def report(self, *losses):
        assert len(losses) == self._n_reported
        for i in range(self._n_reported):
            self._raw[i].append(assert_numpy(losses[i]))

    def print(self, *losses):
        iter = len(self._raw[0])
        report_str = '{0}\t'.format(iter)
        for i in range(self._n_reported):
            report_str += "{0}: {1:.4f}\t".format(self._labels[i], self._raw[i][-1])
        print(report_str)

    def losses(self, n_smooth=1):
        x = np.arange(n_smooth, len(self._raw[0])+1)
        ys = []
        for (label, raw) in zip(self._labels, self._raw):
            raw = assert_numpy(raw).reshape(-1)
            kernel = np.ones(shape=(n_smooth,)) / n_smooth
            ys.append(np.convolve(raw, kernel, mode="valid"))
        return self._labels, x, ys

    def recent(self, n_recent=1):
        return np.array([raw[-n_recent:] for raw in self._raw])


class KLTrainer(object):

    def __init__(self, bg, optim=None, train_likelihood=True, train_energy=True):
        """ Trainer for minimizing the forward or reverse

        Trains in either of two modes, or a mixture of them:
        1. Forward KL divergence / energy based training. Minimize KL divergence between
           generation probability of flow and target distribution
        2. Reverse KL divergence / maximum likelihood training. Minimize reverse KL divergence between
           data mapped back to latent space and prior distribution.

        """
        self.bg = bg

        if optim is None:
            optim = torch.optim.Adam(bg.parameters(), lr=5e-3)
        self.optim = optim

        loss_names = []
        self.train_likelihood = train_likelihood
        self.w_likelihood = 0.0
        self.train_energy = train_energy
        self.w_energy = 0.0
        if train_likelihood:
            loss_names.append("NLL")
            self.w_likelihood = 1.0
        if train_energy:
            loss_names.append("KLL")
            self.w_energy = 1.0
        self.reporter = LossReporter(*loss_names)


    def train(self, n_iter, data=None, batchsize=128, w_likelihood=None, w_energy=None, n_print=0):
        if w_likelihood is None:
            w_likelihood = self.w_likelihood
        if w_energy is None:
            w_energy = self.w_energy

        for iter in range(n_iter):
            self.optim.zero_grad()
            reports = []

            if self.train_likelihood:
                N = data.shape[0]
                idxs = np.random.choice(N, size=batchsize, replace=True)
                batch = data[idxs]

                # negative log-likelihood of the batch is equal to the energy of the BG
                nll = self.bg.energy(batch).mean()
                reports.append(nll)
                # aggregate weighted gradient
                if w_likelihood > 0:
                    l = w_likelihood / (w_likelihood + w_energy)
                    (l * nll).backward(retain_graph=True)
            if self.train_energy:
                # kl divergence to the target
                kll = self.bg.kldiv(batchsize).mean()
                reports.append(kll)
                # aggregate weighted gradient
                if w_energy > 0:
                    l = w_energy / (w_likelihood + w_energy)
                    (l * kll).backward(retain_graph=True)

            self.reporter.report(*reports)
            if n_print > 0:
                if iter % n_print == 0:
                    self.reporter.print(*reports)

            self.optim.step()

    def losses(self, n_smooth=1):
        return self.reporter.losses(n_smooth=n_smooth)
    
    
class SteinTrainer(object):

    def __init__(self, sg, train_lre=False, lrenet=None, train_energy=False, energynet=None, train_score=False, scorenet=None, train_stein=False, scorediff=None, Goptim=None, Doptim=None):
        """ Trainer for minimizing the forward or reverse
        Trains in either of two modes, or a mixture of them:
        1. Forward KL divergence / energy based training. Minimize KL divergence between
           generation probability of flow and target distribution
        2. Reverse KL divergence / maximum likelihood training. Minimize reverse KL divergence between
           data mapped back to latent space and prior distribution.
        """
        self.G = sg.G
        self.D = None


        if train_lre:
            self.D = lrenet
        elif train_energy:
            self.D = energynet
        elif train_score:
            self.D = scorenet
        elif train_stein:
            self.D = scorediff
        else:
            raise ValueError('no training option')

        assert (self.D is not None)


        print(self.D)



        if Goptim is None:
            self.Goptim = torch.optim.Adam(self.G.parameters(), lr=5e-3)
        else:
            self.Goptim = Goptim

        if Doptim is None:
            self.Doptim = torch.optim.Adam(self.D.parameters(), lr=5e-3)
        else:
            self.Doptim = Doptim

        loss_names = []
        self.train_lre = train_lre
        self.train_energy = train_energy
        self.train_score = train_score
        self.train_stein = train_stein

        if train_lre:
            loss_names.append("surKLdiv")
        if train_energy:
            loss_names.append("KLdiv")
        if train_score:
            loss_names.append("surFISHdiv")
        if train_stein:
            loss_names.append("FISHdiv")

        self.reporter = LossReporter(*loss_names)


    def train(self, n_iter, data=None, batchsize=128, n_print=0):

        for iter in range(n_iter):
            self.Goptim.zero_grad()
            self.Doptim.zero_grad()

            reports = []

            if self.train_lre:
                for _ in range(5):
                    self.Doptim.zero_grad()

                    N = data.shape[0]
                    idxs = np.random.choice(N, size=batchsize, replace=True)
                    images = data[idxs].cuda()

                    images = images.view(images.shape[0], -1)
                    z = torch.rand(batchsize, 2).to(images.device)

                    real_labels = torch.ones(batchsize).cuda()
                    fake_labels = torch.zeros(batchsize).cuda()

                    outputs = self.D(images)
                    d_loss_real = nn.BCELoss()(outputs.flatten(), real_labels)
                    real_score = outputs

                    # Compute BCELoss using fake images
                    fake_images = self.G(z)
                    outputs = self.D(fake_images)
                    d_loss_fake = nn.BCELoss()(outputs.flatten(), fake_labels)
                    fake_score = outputs

                    # Optimizie discriminator
                    d_loss = d_loss_real + d_loss_fake
                    d_loss.backward()
                    self.Doptim.step()
                

                self.Goptim.zero_grad()
                z = torch.rand(batchsize, 2).to(images.device)
                fake_images = self.G(z)
                outputs = self.D(fake_images)
                g_loss = nn.BCELoss()(outputs.flatten(), real_labels)

                g_loss.backward()
                self.Goptim.step()
                reports.append(g_loss.mean())


            if self.train_energy:
                raise NotImplementedError('Not implemented')
            if self.train_stein:
                raise NotImplementedError('Not implemented')
            if self.train_score:
                raise NotImplementedError('Not implemented')       


            self.reporter.report(*reports)
            if n_print > 0:
                if iter % n_print == 0:
                    self.reporter.print(*reports)

    def losses(self, n_smooth=1):
        return self.reporter.losses(n_smooth=n_smooth)
