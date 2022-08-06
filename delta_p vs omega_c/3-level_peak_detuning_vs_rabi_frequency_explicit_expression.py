#IMPORT MODULES-----------------------------------------------------------
import numpy as np
from numpy.fft import fft, ifft, fftshift, ifftshift

import matplotlib.pyplot as plt

import scipy as sp
from scipy import special
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
from scipy.signal import find_peaks
import scipy.optimize as opt

import time

plt.style.use(['science', 'notebook', 'grid'])

#START TIME----------------------------------------------------------------
start = time.time()

#FUNCTIONS-----------------------------------------------------------------
def delta_peak1t(B):
    return (1/4)*(np.sqrt(2)*np.sqrt(A**2 + np.sqrt(A**2 - 4)*A + 8*B - 2) + np.sqrt(A**2 - 4) + A)

def delta_peak2t(B):
    return (1/4)*(np.sqrt(2)*np.sqrt(A**2 + np.sqrt(A**2 - 4)*A + 8*B - 2) - np.sqrt(A**2 - 4) - A)

def delta_peak1(B):
    return (1/2)*(A + np.sqrt(A**2 + 4*B)) #type I

def delta_peak2(B):
    return (1/2)*(-A + np.sqrt(A**2 + 4*B)) #type II

#CONSTANTS-----------------------------------------------------------------
#Dephasing Rate between |1> (|g>) and |3> (|e>) (rad/sec)
gamma13 = 1.8850e7

#Dephasing Rate between |1> (|g>) and |2> (|c>) (rad/sec)
gamma12 = 0.005*gamma13

#Optical Depth (Resonant, Non-Zero Temperature)
bv0 = 62

#Material Thickness (m)
L = 0.017

#resonant absorption coefficient
alpha = bv0/L

A = alpha*L/(2*np.pi)

OmegaCNorm = np.linspace(0, 10, 20)
B = OmegaCNorm**2

#PLOTTING------------------------------------------------------------------
fig, ax = plt.subplots(1, 1, figsize = (12.7, 9.7))

ax.plot(OmegaCNorm, delta_peak1t(B), color = "red")
ax.plot(OmegaCNorm, delta_peak2t(B), color = "blue")
ax.plot(OmegaCNorm, delta_peak1(B), color = "cyan")
ax.plot(OmegaCNorm, delta_peak2(B), color = "yellow")

ax.set_title("Peak Detuning, $\Delta_{peak}$ vs Rabi Frequency, $\Omega_{c}$", fontsize = 31)

#Label axes
ax.set_xlabel("$\Omega_{c}/\gamma_{13}$", fontsize = 29)
ax.set_ylabel("$\Delta_{peak}/\gamma_{13}$", fontsize = 29)

#modify ticks
ax.tick_params(axis = "x", labelsize = 29)
ax.tick_params(axis = "y", labelsize = 29)

#Enable Legend
ax.legend(fontsize = 29)

#adjust subplots
fig.tight_layout()

#END TIME-----------------------------------------------------------------
#sleeping for 1 sec to get 10 sec runtime
time.sleep(1)

#end time
end = time.time()

#TOTAL RUNTIME------------------------------------------------------------
#total time taken
print(f"Runtime of the program is {end - start}")