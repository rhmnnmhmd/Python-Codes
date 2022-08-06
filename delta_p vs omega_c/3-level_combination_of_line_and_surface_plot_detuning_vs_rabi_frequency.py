##IMPORT MODULES##
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import special
from numpy.fft import fft, ifft, fftshift, ifftshift
from scipy.optimize import curve_fit
import scipy.optimize as opt
import time
from scipy.signal import argrelextrema
from scipy.signal import find_peaks

plt.style.use(['science', 'notebook', 'grid'])

#START TIME---------------------------------------------------------------
start = time.time()

#-----------------------------------------------------------------------------
##FUNCTION DEFINITIONS##
#Eletric Susceptibility
def susc(wNew, m):
    part1 = 4*(wNew + 1j*gamma12)*gamma13
    part2 = OmegaCArray[m]**2 - 4*(wNew + 1j*gamma12)*(wNew + 1j*gamma13)
    return ((c*bv0)/(w0*L))*part1/part2

#Refractive Index
def eta(wNew, m):
    return np.sqrt(1 + susc(wNew, m))

#Wavenumber
def k(wNew, m):
    return (w0/c)*eta(wNew, m)

#fitting function
def func1(OmegaCArray, a, b, c):
    return a*OmegaCArray**2 + b*OmegaCArray + c

def func2(OmegaCArray, d, e, f):
    return d*OmegaCArray**2 + e*OmegaCArray + f

def func3(OmegaCArray, g, h, i):
    return g*OmegaCArray**2 + h*OmegaCArray + i

#Real g-function
def g(x):
    return np.sqrt(np.pi/8)*(1/x)*special.wofz(1j/(np.sqrt(8)*x))
#-----------------------------------------------------------------------------
##CONSTANTS##
#Resonant Wavelength (m)
lamda0 = 7.95e-7
#value from:
#Du, S., Kolchin, P., Belthangady, C., Yin, G. Y., & Harris, S. E. (2008). 
#Subnatural linewidth biphotons with controllable temporal length. Physical Review Letters, 100(18). 
#https://doi.org/10.1103/PhysRevLett.100.183603

#Resonant Wavenumber (1/m)
k0 = (2*np.pi)/(lamda0)

#Vacuum Speed of Light (m/sec)
c = 3e8

#Resonant Angular Frequency (rad/sec)
w0 = k0*c

#Dephasing Rate between |1> (|g>) and |3> (|e>) (rad/sec)
gamma13 = 1.8850e7

#Dephasing Rate between |1> (|g>) and |2> (|c>) (rad/sec)
gamma12 = 0.005*gamma13
gamma21 = gamma12
gamma21bar = gamma21/gamma13

#OD
bv0 = 62
    
#Scattering/Absorption Cross-Section (m^2)
sigma = (3*(lamda0**2))/(2*np.pi)

#Material Thickness (m)
L = 0.017

alpha = bv0/L

A = alpha*L / (2*np.pi)
#-----------------------------------------------------------------------------
##SAMPLING PARAMETERS##
#Sampling Rate/Points
N = 80000   

#Time-Domain Full Interval
T = 2.5e-6

#Sampling Interval
dt = T/N
#-----------------------------------------------------------------------------
##INPUT ARRAY##
#t-domain Array
t = np.linspace(0, T, N)
#-----------------------------------------------------------------------------    
#Ignition Time
tUp = 0.125e-6

#Extinction Time
tDown = 2e-6

#Input Pulse at z = 0
E00 = np.heaviside(t - tUp, 1) - np.heaviside(t - tDown, 1)

#Input Pulse at z = L
E0L = E00*np.exp(1j*k0*L)

#FFT of E00
E00f = fft(E00)

#FFT of E0L
E0Lf = fft(E0L)

#rotated/shifted E00f
E00fr = fftshift(E00f)  

#Rotated/Shifted E0Lf 
E0Lfr = fftshift(E0Lf)   

#-------------------------------------------------------------------------
#detuningArray = np.array([])
N1 = 100
detuningArray = np.linspace(0*gamma13, 20*gamma13, N1)
detuningArrayNorm = detuningArray/gamma13

#empty b0 array
N2 = 100
OmegaCArray = np.linspace(0*gamma13, 20*gamma13, N2)
OmegaCArrayNorm = OmegaCArray/gamma13
OmegaCArrayNorm2 = np.linspace(0, 20, 20)

#Empty I_s Array
IsArray = np.zeros((N1, N2))

#Angular Frequency
f = np.linspace(-N/(2*T), N/(2*T), N)
w = (2.0*np.pi)*f

#Loop to Fill Isarray
for m in range(0, N2):
    for n in range(0, N1):
        #Detuned Angular Frequency
        wNew = [i + detuningArray[n] for i in w]
            
        #Change dNew to array-type
        wNew = np.asarray(wNew)
        
        #Transfer Function
        trans = np.exp(1j*k(wNew, m)*L)  
            
        #Frequency-Domain Transmitted Field
        Efr = E0Lfr*trans
            
        #IFFT of Frequency-Domain Transmitted Field
        Ef = ifftshift(Efr)
            
        #Time-Domain Transmitted Field
        E = ifft(Ef)
            
        #Finding Element Index Where Probe Ignition Occured
        for i in t:
            if i >= tUp:
                upIndex = np.where(t == i)
                upIndex = upIndex[0][0]
                break
        
        #Finding Element Index Where Probe Extinction Occurs
        for i in t:
            if i >= tDown:
                downIndex = np.where(t == i)
                downIndex = downIndex[0][0]
                break
        
        #Front part of incident pulse before probe ignition
        part1 = np.zeros(upIndex)
            
        #size of part1
        part1Size = len(part1)
        
        #Datapoints That is Left After Front Part has Been Chosen
        sizeLeft = N - part1Size
        
        #Size of Time-Domain Transmitted Field From t= 0 Up Till Probe Extinction
        size = len(E[0: downIndex])
        
        #If-Else Statements to Determine The Final Time-Domain Transmitted Field
        if size < sizeLeft:
            part2 = np.flip(np.copy(E[0: downIndex]))
            part2Size = len(part2)
            sizeLeft = N - part1Size - part2Size
            sizeLeft = -sizeLeft
            part3 = np.flip(np.copy(E[sizeLeft: ]))
            Efinal = np.append(part1, part2)
            Efinal = np.append(Efinal, part3)
        elif size == sizeLeft:
            part2 = np.flip(np.copy(E[0: downIndex]))
            Efinal = np.append(part1, part2)  
        elif size > sizeLeft:
            sizeLeft = upIndex + downIndex - N
            part2 = np.flip(np.copy(E[sizeLeft: downIndex]))
            Efinal = np.append(part1, part2)
        else:
            print("Error!!!")
            
        #Time-Domain Transmitted intensity
        It = np.abs(Efinal)**2
            
        #Finding Max Time-Domain Transmitted Intensity
        Is = np.max(It[downIndex: downIndex+200])
            
        #Appending Isarray
        IsArray[n, m] = Is

x = np.linspace(0, 20, 200)
y = np.linspace(0, 20, 200)
X, Y = np.meshgrid(x, y)
phibar = 0.1
phibar2 = -0.1
exactPhase1 = (phibar-1)*Y**4 + A*Y**3 + ((phibar-1)*(1+gamma21bar)**2 + 2*(1-phibar)*(0.25*X**2 + gamma21bar))*Y**2 - A*(0.25*X**2 - gamma21bar**2)*Y + (phibar-1)*(0.25*X**2 + gamma21bar)**2
exactPhase2 = (phibar2+1)*Y**4 + A*Y**3 + ((phibar2+1)*(1+gamma21bar)**2 - 2*(1+phibar2)*(0.25*X**2 + gamma21bar))*Y**2 - A*(0.25*X**2 - gamma21bar**2)*Y + (phibar2+1)*(0.25*X**2 + gamma21bar)**2

for m in range(200):
    for n in range(200):
        
        if (m < 100):
            exactPhase1[m, n] = 5000
        else:
            None
            
for m in range(200):
    for n in range(200):
        
        if (m > 0.4*n):
            exactPhase2[m, n] = -5000
        else:
            None

exactPhase1[Y < y[100]] = 5000
exactPhase2[Y > 0.4*X] = -5000

#PLOTTING----------------------------------------------------------------------
#create figure and axis
fig, ax = plt.subplots(1, 1, figsize=(12.7, 9.7))

#numerical data for type 1 and type 2
ax.plot(OmegaCArrayNorm2, avg1/gamma31, "ok", label = "Data (Type 1)", markersize = 10)
ax.plot(OmegaCArrayNorm2, avg3/gamma31, "^k", label = "Data (Type 2)", markersize = 10)

#Numerical data for minimas
#ax.plot(OmegaCArrayNorm2, SpecWidthMinArray1_000/gamma31, "sk", label = "Data (Min Type 1)")
#ax.plot(OmegaCArrayNorm2, SpecWidthMinArray2_000/gamma31, "*k", label = "Data (Min Type 2)")

#Numerical data for local maxima
#ax.plot(OmegaCArrayNorm2, SpecWidthAltMaxArray1/gamma31, "Dk", label = "Data (Alt Type 1)", markersize = 10)
#ax.plot(OmegaCArrayNorm2, SpecWidthAltMaxArray2/gamma31 + 0.7, "Pk", label = "Data (Alt Type 2)", markersize = 10)

##LINE PLOTS##
CS1 = ax.contour(X, Y, exactPhase1, [0], colors = "red", linestyles = "solid")
CS2 = ax.contour(X, Y, exactPhase2, [0], colors = "blue", linestyles = "dashed")
#ax.clabel(CS1, inline=1, fontsize=15)
#ax.clabel(CS2, inline=1, fontsize=10)
labels = ["Eq. (8) (Type 1)", "Eq. (8) (Type 2)"]
CS1.collections[0].set_label(labels[0])
CS2.collections[0].set_label(labels[1])

level1 = CS1.collections[0]
#del(level0.get_paths()[0])

level2 = CS2.collections[0]
        
#Label axes
ax.set_xlabel("$\Omega_{c}/\gamma_{31}$", fontsize = 29)
ax.set_ylabel("$\Delta_{p}/\gamma_{31}$", fontsize = 29)

#modify ticks
ax.tick_params(axis = "x", labelsize = 29)
ax.tick_params(axis = "y", labelsize = 29)

#Enable Legend
ax.legend(loc = "upper left", fontsize = 19)
ax.grid(False)


#SURFACE PLOTS
#create domain grid
OmegaCGrid, detuningGrid = np.meshgrid(OmegaCArrayNorm, detuningArrayNorm)

#create the plot itself
cs = ax.contourf(OmegaCGrid, detuningGrid, IsArray, cmap = "hot", vmin = 0, vmax = 4, levels = 100)

#generate colorbar indicator
colBar = fig.colorbar(cs)
colBar.set_label(label = "$I_{s}/I_{0}$", size = 29)                
colBar.ax.tick_params(labelsize = 29)

fig.tight_layout()

#END TIME-----------------------------------------------------------------------------
#sleeping for 1 sec to get 10 sec runtime
time.sleep(1)

#end time
end = time.time()

##TOTAL RUNTIME##
#total time taken
print(f"Runtime of the program is {end - start}")