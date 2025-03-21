#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from qiskit.visualization import array_to_latex
from scipy.linalg import sqrtm


# In[ ]:


# ===== RBM 클래스 정의 =====
class RBM(nn.Module):
    def __init__(self, n_visible, n_hidden):
        super(RBM, self).__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.W = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01)
        self.a = nn.Parameter(torch.zeros(n_visible))
        self.b = nn.Parameter(torch.zeros(n_hidden))

    def sample_h(self, v):
        p_h = torch.sigmoid(torch.matmul(v, self.W) + self.b)
        return p_h, torch.bernoulli(p_h)

    def sample_v(self, h):
        p_v = torch.sigmoid(torch.matmul(h, self.W.t()) + self.a)
        return p_v, torch.bernoulli(p_v)

    def free_energy(self, v):
        wx_b = torch.matmul(v, self.W) + self.b
        vbias_term = torch.matmul(v, self.a)
        hidden_term = torch.sum(torch.log1p(torch.exp(wx_b)), dim=1)
        return -vbias_term - hidden_term

# ===== Ideal state 샘플 생성 함수 (computational basis 샘플) =====
def generate_ideal_state_samples(n_qubits, n_samples, ideal_state):
    probs = np.abs(ideal_state)**2
    probs = probs / np.sum(probs)
    indices = np.random.choice(len(ideal_state), size=n_samples, p=probs)
    data = []
    for idx in indices:
        binary_str = format(idx, '0' + str(n_qubits) + 'b')
        sample = [int(bit) for bit in binary_str]
        data.append(sample)
    return torch.tensor(data, dtype=torch.float)

# ===== RBM 학습 함수 (Contrastive Divergence) =====
def train_rbm(rbm, data, epochs, batch_size, lr, k):
    optimizer = optim.SGD(rbm.parameters(), lr=lr)
    n_samples = data.shape[0]
    losses = []
    for epoch in range(epochs):
        perm = torch.randperm(n_samples)
        epoch_loss = 0.0
        for i in range(0, n_samples, batch_size):
            batch = data[perm[i:i+batch_size]]
            p_h, h_sample = rbm.sample_h(batch)
            v_k = batch.clone()
            for _ in range(k):
                p_h_k, h_k = rbm.sample_h(v_k)
                p_v_k, v_k = rbm.sample_v(h_k)
            loss = torch.mean(rbm.free_energy(batch)) - torch.mean(rbm.free_energy(v_k))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.shape[0]
        epoch_loss /= n_samples
        losses.append(epoch_loss)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
    return losses

def all_states(n):
    states = np.array(np.meshgrid(*[[0, 1]] * n)).T.reshape(-1, n)
    return torch.tensor(states, dtype=torch.float)


# In[2]:


def plot_rho(rho, title, vmax=None, default_vmax=0.5):
    # 자동 vmax 설정: rho의 값이 default_vmax보다 작으면 default_vmax 사용
    auto_vmax = max(
        np.max(np.abs(rho.real)),
        np.max(np.abs(rho.imag))
    )

    if vmax is None:
        vmax = default_vmax if auto_vmax < default_vmax else auto_vmax

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    xpos, ypos = np.meshgrid(np.arange(rho.shape[0]), np.arange(rho.shape[1]), indexing="ij")
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros_like(xpos)
    dz = rho.flatten()

    colors = ['#A0522D' if val < 0 else '#FFD700' for val in dz.real]

    ax.bar3d(xpos, ypos, zpos, 0.5, 0.5, dz.real, color=colors, shade=True)

    ax.set_zlim(-vmax, vmax)
    ax.set_xlabel('Row Index')
    ax.set_ylabel('Column Index')
    ax.set_zlabel('Value')
    ax.set_title(title)
    plt.show()


# In[3]:


def fidelity(rho, sigma):
    """
    Compute the quantum fidelity between two density matrices.

    Parameters:
        rho (ndarray): Ideal density matrix
        sigma (ndarray): Reconstructed density matrix

    Returns:
        float: Fidelity value (0 ≤ F ≤ 1)
    """
    # Compute sqrt of rho using matrix square root
    sqrt_rho = sqrtm(rho)

    # Compute sqrt_rho * sigma * sqrt_rho
    inner_term = sqrt_rho @ sigma @ sqrt_rho

    # Compute square root of the inner term
    sqrt_inner = sqrtm(inner_term)

    # Compute trace and square it
    fid = np.real(np.trace(sqrt_inner)) ** 2

    # Ensure fidelity is within [0, 1]
    return print(min(fid, 1.0))

def purity(rho):
    """
    주어진 밀도 행렬(rho)에 대한 Purity 계산
    :param rho: 밀도 행렬 (numpy.ndarray)
    :return: Purity 값 (float)
    """
    # Purity 계산: Tr(ρ^2)
    purity = np.trace(np.matmul(rho, rho)).real
    return print(purity)


# In[ ]:




