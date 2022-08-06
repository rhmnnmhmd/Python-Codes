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
from scipy.special import wofz

plt.style.use(['science', 'notebook', 'grid'])

#START TIME---------------------------------------------------------------
start = time.time()

#-----------------------------------------------------------------------------
##FUNCTION DEFINITIONS##
#Eletric Susceptibility
def trans(wNew, m):
    x = (wNew + (1.0j*gamma/2))/(np.sqrt(2)*k0*vbar)
    return np.exp((-bv0Array[m]/(2*g(3.4)))*np.sqrt(np.pi/8)*(gamma/(k0*vbar))*wofz(x))

#Real g-function
def g(x):
    return np.sqrt(np.pi/8)*(1/x)*special.wofz(1j/(np.sqrt(8)*x))
#-----------------------------------------------------------------------------
##CONSTANTS##
#Resonant Wavelength (m)
lamda0 = 6.89e-7
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

gamma = 4.7124e4

vbar = 3.4*gamma/k0

#medium peak density (m^-3)
rho = 4.6e17

#optical thickness (nonzero temperature, resonant)
#bv0 = 19

#optical thickness (0 temperature, resonant)
#b00 = bv0/g(3.4)            #kwong thesis eqn 3.67

#scattering/absorption cross-section
sigma = (3*(lamda0**2))/(2*np.pi)

L = 0.0011075568414104576

#absorption coefficient
#alpha = bv0/L

#-----------------------------------------------------------------------------
##SAMPLING PARAMETERS##
#Sampling Rate/Points
N = 80000   

#Time-Domain Full Interval
T = 8e-5

#Sampling Interval
dt = T/N
#-----------------------------------------------------------------------------
##INPUT ARRAY##
#t-domain Array
t = np.linspace(0, T, N)
#-----------------------------------------------------------------------------    
#Ignition Time
tUp = 1.0e-5

#Extinction Time
tDown = 5.0e-5

#Input Pulse at z = 0
E00 = np.heaviside(t - tUp, 1) - np.heaviside(t - tDown, 1)
#fft of incident field at z = 0
E00f = fft(E00)
E00fr = fftshift(E00f)

#t-domain incident field at z = L
E0L = E00*np.exp(1.0j*k0*L)
#fft of incident field at z = L
E0Lf = fft(E0L)
E0Lfr = fftshift(E0Lf) 

#-------------------------------------------------------------------------
#detuningArray = np.array([])
N1 = 453
detuningArray = np.linspace(0*gamma, 140*gamma, N1)
detuningArrayNorm = detuningArray/gamma

#empty b0 array
N2 = 100
bv0Array = np.linspace(1, 200, N2)
b00Array = bv0Array/g(3.4)

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
            
        #Frequency-Domain Transmitted Field
        Efr = E0Lfr*trans(wNew, m)
            
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

#PLOTTING----------------------------------------------------------------------
#IsArrayPlotting = IsArray*10000

#create domain grid
bv0Grid, detuningGrid = np.meshgrid(bv0Array, detuningArrayNorm)

#create figure and axis
fig, ax = plt.subplots(1, 1, figsize=(12.7, 9.7))

#create the plot itself
cs = ax.contourf(bv0Grid, detuningGrid, IsArray, cmap = "hot", vmin = 0, vmax = 4, levels = 50)

#set ylim
#ax.set_ylim(0, 2.5)

#generate colorbar indicator
colBar = fig.colorbar(cs)

ax.set_xlabel("$b_{0}$")
ax.set_ylabel("$\Delta/\Gamma$")

# x = np.linspace(0, 10, 50)

# y = np.linspace(0, 5, 25)

# X, Y = np.meshgrid(x, y)

# Z = X**2 + Y**2

# fig, ax = plt.subplots(1, 1, figsize=(12.7, 9.7))

# #create the plot itself
# cs = ax.contourf(X, Y, Z, cmap = "hot")

#END TIME-----------------------------------------------------------------------------
#sleeping for 1 sec to get 10 sec runtime
time.sleep(1)

#end time
end = time.time()

##TOTAL RUNTIME##
#total time taken
print(f"Runtime of the program is {end - start}")