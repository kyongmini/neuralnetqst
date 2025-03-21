#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


def similarity_transformation(theta):
    """
    Similarity transformation for rotation of Jones matrix
    """
    R_matrix = np.array([(np.cos(theta), -np.sin(theta)),
                  (np.sin(theta), np.cos(theta))])
    return R_matrix


# In[3]:


def polarization_analysis(h, q):
    """
    QWP - HWP - PBS
    select the measurement basis by choosing h, q varialbes.
    H = (Polarization_Analysis(0, 0))
    V = 1j*Polarization_Analysis(np.pi/4, 0)
    D = (Polarization_Analysis(np.pi/8, np.pi/4))
    A = (Polarization_Analysis(-np.pi/8, -np.pi/4))
    L = (Polarization_Analysis(-np.pi/8, 0))
    R = (Polarization_Analysis(np.pi/8, 0))
    """
    # Compute the transformed Jones matrices for HWP and QWP
    U_HWP = Similarity_Transformation(h) @ HWP @ np.linalg.inv(Similarity_Transformation(h))
    U_QWP = Similarity_Transformation(q) @ QWP @ np.linalg.inv(Similarity_Transformation(q))

    # inverse tracking of QWP - HWP - PBS
    projection_state = np.linalg.inv(U_QWP) @ np.linalg.inv(U_HWP) @ PBS_t_state
    return projection_state

