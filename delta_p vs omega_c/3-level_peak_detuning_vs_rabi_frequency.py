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

import sympy as smp
from sympy import symbols, plot_implicit, Eq, And, lambdify
from sympy.plotting import plot_parametric

import time

import pickle

plt.style.use(['science', 'notebook', 'grid'])

with open('data_4.pkl', 'rb') as file:
    SpecWidthNMaxArray = pickle.load(file)

#START TIME---------------------------------------------------------------
start = time.time()

#FUNCTION DEFINITIONS-----------------------------------------------------
#Eletric Susceptibility
def susc(wNew):
    part1 = 4*(wNew + 1j*gamma21)*gamma31
    part2 = OmegaC**2 - 4*(wNew + 1j*gamma21)*(wNew + 1j*gamma31)
    return (c*alpha/w0)*part1/part2

#Refractive Index
def n(wNew):
    return np.sqrt(1 + susc(wNew))

#Wavenumber
def k(wNew):
    return (w0/c)*n(wNew)

#Real g-function
def g(x):
    return np.sqrt(np.pi/8)*(1/x)*special.wofz(1j/(np.sqrt(8)*x))

#quadratic fitting function
def func1(OmegaCArray, a, b, c):
    return a*OmegaCArray**2 + b*OmegaCArray + c

#quartic fitting function 
def func2(OmegaCArray, d, e, f, g, h):
    return d*OmegaCArray**4 + e*OmegaCArray**3 + f*OmegaCArray**2 + g*OmegaCArray + h

#theory expression from SymPy (type 1, appoximated)
def d1(OmegaCArrayNorm):
    C = np.sqrt(A**2 - 4) + A
    return np.sqrt((OmegaCArrayNorm**2)/4 + (C**2)/16) + C/4

#theory expression from SymPy (type 2, approximated)
def d2(OmegaCArrayNorm):
    C = np.sqrt(A**2 - 4) + A
    return np.sqrt((OmegaCArrayNorm**2)/4 + (C**2)/16) - C/4
    
# #Theory expression from Mathematica (type 1, approximated)
# def delta_peak1(OmegaCArrayNorm):
#     return (1/4)*(np.sqrt(2)*np.sqrt(A**2 + np.sqrt(A**2 - 4)*A + 2*OmegaCArrayNorm**2 - 2) + np.sqrt(A**2 - 4) + A)

# ##Theory expression from Mathematica (type 2, approximated)
# def delta_peak2(OmegaCArrayNorm):
#     return (1/4)*(np.sqrt(2)*np.sqrt(A**2 + np.sqrt(A**2 - 4)*A + 2*OmegaCArrayNorm**2 - 2) - np.sqrt(A**2 - 4) - A)

#CONSTANTS----------------------------------------------------------------
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
gamma31 = 1.8850e7

#Dephasing Rate between |1> (|g>) and |2> (|c>) (rad/sec)
gamma21 = 0.005*gamma31
gamma21bar = gamma21/gamma31
C = gamma21bar**2

#Optical Depth (Resonant, Non-Zero Temperature)
bv0 = 62

#Scattering/Absorption Cross-Section (m^2)
sigma = (3*(lamda0**2))/(2*np.pi)

#Material Thickness (m)
L = 0.017

#resonant absorption coefficient
alpha = bv0/L

A = alpha*L/(2*np.pi)

#SAMPLING PARAMETERS------------------------------------------------------
#Sampling Rate/Points
N = 80000   

#Time-Domain Full Interval
T = 2.5e-6

#Sampling Interval
dt = T/N

#INPUT ARRAY##------------------------------------------------------------
#t-domain Array
t = np.linspace(0, T, N)

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

#SPECTRAL WIDTH VS OMEGAC STUFF-------------------------------------------
#x-axis array
OmegaCArray = np.linspace(0, 20*gamma31, 20)

#normalize OmegaCArray
OmegaCArrayNorm = OmegaCArray/gamma31
OmegaCArrayNormAlt = OmegaCArrayNorm/2

#y-axis for global max
SpecWidthMaxArray = np.array([])

#y-axis for global min
SpecWidthMinArray = np.array([])

#y-axis for near-resonance max
SpecWidthNMaxArray = np.array([])

#y-axis for alternate maxima 1
SpecWidthAltMaxArray1 = np.array([])

#y-axis for alternate maxima 2
SpecWidthAltMaxArray2 = np.array([])

#empty array for phases of type 1
#phaseChosen1Array = np.array([])

#empty array for phases of type 2
#phaseChosen2Array = np.array([])

#Detuning Array
detuningArray = np.arange(-30*gamma31, 30*gamma31, 2.5e6)

#Normalize detuning_array w.r.t. gamma31
detuningArrayNorm = detuningArray/gamma31

#Angular Frequency
f = np.linspace(-N/(2*T), N/(2*T), N)
w = (2.0*np.pi)*f  

#loop to fill spectral width arrays
for OmegaCArrayElement in OmegaCArray:
    ##GENERATE I_S AND DETUNING ARRAYS##
    #Empty I_s Array
    IsArray = np.array([])
    #Loop to Fill Isarray
    for detuningArrayElement in detuningArray:
        #Detuned Angular Frequency
        wNew = [wElement + detuningArrayElement for wElement in w]
        
        #Change dNew to array-type
        wNew = np.asarray(wNew)
        
        #Transfer Function
        OmegaC = OmegaCArrayElement
        trans = np.exp(1j*k(wNew)*L)  
        
        #for phase-shift plot
        # phase = k(wNew)*L
        # phaseShift = phase - k0*L
        # phaseShiftNorm = phaseShift/np.pi
        
        #Frequency-Domain Transmitted Field
        Efr = E0Lfr*trans
        
        #IFFT of Frequency-Domain Transmitted Field
        Ef = ifftshift(Efr)
        
        #Time-Domain Transmitted Field
        E = ifft(Ef)
        
        #Finding Element Index Where Probe Ignition Occured
        for a in t:
            
            if a >= tUp:
                upIndex = np.where(t == a)
                upIndex = upIndex[0][0]
                
                break
    
        #Finding Element Index Where Probe Extinction Occurs
        for b in t:
            
            if b >= tDown:
                downIndex = np.where(t == b)
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
        IsArray = np.append(IsArray, Is)
    
    #midpoint index for detuning array   
    midPointIndex = len(detuningArray)/2
    midPointIndex = round(midPointIndex)
    
    #Finding maximal detuning
    maxValue = np.max(IsArray[midPointIndex: ])
    maxValueIndex = np.where(IsArray == maxValue)
    maxValueIndexArr = maxValueIndex[0]
    maxValueIndexInt = maxValueIndexArr[0]
    maxDetuning = detuningArray[maxValueIndexInt]
    
    #finding minimal detuning
    minIndex, _ = find_peaks(-1*IsArray[midPointIndex: ], -0.75)
    minIndexFocus = midPointIndex + minIndex[-1]
    minDetuning = detuningArray[minIndexFocus]
    
    #finding near-resonance maximal
    NMaxIndexArray, _ = find_peaks(IsArray[midPointIndex: minIndexFocus])
    
    if len(NMaxIndexArray) == 0:
        NMaxDetuning = 0.0
        
    elif len(NMaxIndexArray) != 0:
        NMaxIndexFocus = midPointIndex + NMaxIndexArray[0]
        NMaxDetuning = detuningArray[NMaxIndexFocus]
        
    else:
        print("ERROR!")

    #finding alternate maximals
    altMaxIndexArray, _ = find_peaks(IsArray[midPointIndex: minIndexFocus])
    
    if len(altMaxIndexArray) == 0:
        altMaxDetuning = 0.0
        
    elif len(altMaxIndexArray) != 0:
        altMaxIndexFocus1 = midPointIndex + altMaxIndexArray[1]
        altMaxIndexFocus2 = midPointIndex + altMaxIndexArray[2]
        altMaxDetuning1 = detuningArray[altMaxIndexFocus1]
        altMaxDetuning2 = detuningArray[altMaxIndexFocus2]
        
    else:
        print("ERROR!")    

    #global maximum
    SpecWidthMax = maxDetuning 
    
    #global minimum
    SpecWidthMin = minDetuning 
    
    #near-resonance maximum
    SpecWidthNMax = NMaxDetuning
    
    #alternate maximas
    SpecWidthAltMax1 = altMaxDetuning1
    SpecWidthAltMax2 = altMaxDetuning2
    
    #create spec width arrays
    #global maximum
    SpecWidthMaxArray = np.append(SpecWidthMaxArray, SpecWidthMax)
    
    #global minimum
    SpecWidthMinArray = np.append(SpecWidthMinArray, SpecWidthMin)
    
    #near-resonance maximum
    SpecWidthNMaxArray = np.append(SpecWidthNMaxArray, SpecWidthNMax)
    
    #alternate maximas
    SpecWidthAltMaxArray1 = np.append(SpecWidthAltMaxArray1, SpecWidthAltMax1)
    SpecWidthAltMaxArray2 = np.append(SpecWidthAltMaxArray2, SpecWidthAltMax2)
    
    #for phase-shift plot
    # OmegaC = OmegaCArrayElement
    # phase = np.real(k(detuningArray))*L
    # phaseShift = phase - k0*L
    # phaseShiftNorm = phaseShift/np.pi
    # phaseChosen1 = phaseShiftNorm[maxValueIndexInt]
    # phaseChosen2 = phaseShiftNorm[NMaxIndexFocus]
    # phaseChosen1Array = np.append(phaseChosen1Array, phaseChosen1)
    # phaseChosen2Array = np.append(phaseChosen2Array, phaseChosen2)

#fitting function using scipy--------------------------------------------------
#quartic fitting parameters
params1, params_cov1 = curve_fit(func2, OmegaCArray, avg1)
params2, params_cov2 = curve_fit(func2, OmegaCArray, avg3)
# params3, params_cov3 = curve_fit(func2, OmegaCArray, SpecWidthAltMaxArray1)
# params4, params_cov4 = curve_fit(func2, OmegaCArray, SpecWidthAltMaxArray2)

#curve fitting
curve_fit1 = func2(OmegaCArray, *params1)/gamma31
curve_fit2 = func2(OmegaCArray, *params2)/gamma31
# curve_fit3 = func2(OmegaCArray, *params3)/gamma31
# curve_fit4 = func2(OmegaCArray, *params4)/gamma31

#approximated, no additional phase shift term----------------------------------
approxNoPhase1 = d1(OmegaCArrayNorm)
approxNoPhase2 = d2(OmegaCArrayNorm)

#approximated, additional phase shift term
#offset = np.abs(d1(0) - func1(0, *params1)/gamma31)
#approxPhase1 = d1(OmegaCArrayNorm) + offset
approxPhase2 = d2(OmegaCArrayNorm)

#for plotting implicit equation using meshgrid style---------------------------
x = np.linspace(0, 20, 200)
y = np.linspace(0, 20, 200)
X, Y = np.meshgrid(x, y)

#approximated, additional phase shift term (implicit plot)
phi = 0.11
approxPhaseImplicit1 = (phi - 1)*Y**4 + A*Y**3 + (phi - 0.5*phi*X**2 - 1 + 0.5*X**2)*Y**2 - 0.25*A*Y*X**2 + 0.0625*(phi - 1)*X**4
approxPhaseImplicit2 = (phi + 1)*Y**4 + A*Y**3 + (phi - 0.5*phi*X**2 + 1 - 0.5*X**2)*Y**2 - 0.25*A*Y*X**2 + 0.0625*(phi + 1)*X**4

#exact, no additional phase shift term
exactNoPhase1 = Y**4 - A*Y**3 + (gamma21bar**2 - 0.5*X**2 + 1)*Y**2 + A*(0.25*X**2 - gamma21bar**2)*Y + gamma21bar**2 + 0.5*gamma21bar*X**2 + 0.0625*X**4
exactNoPhase2 = Y**4 + A*Y**3 + (gamma21bar**2 - 0.5*X**2 + 1)*Y**2 - A*(0.25*X**2 - gamma21bar**2)*Y + gamma21bar**2 + 0.5*gamma21bar*X**2 + 0.0625*X**4

#exact, additional phase shift term
phibar = 0.1
phibar2 = -0.1
exactPhase1 = (phibar-1)*Y**4 + A*Y**3 + ((phibar-1)*(1+gamma21bar)**2 + 2*(1-phibar)*(0.25*X**2 + gamma21bar))*Y**2 - A*(0.25*X**2 - gamma21bar**2)*Y + (phibar-1)*(0.25*X**2 + gamma21bar)**2
exactPhase2 = (phibar2+1)*Y**4 + A*Y**3 + ((phibar2+1)*(1+gamma21bar)**2 - 2*(1+phibar2)*(0.25*X**2 + gamma21bar))*Y**2 - A*(0.25*X**2 - gamma21bar**2)*Y + (phibar2+1)*(0.25*X**2 + gamma21bar)**2
exactPhase3 = (1 + phibar2)*Y**4 + A*Y**3 + (1+phibar2)*((1+gamma21bar)**2 - 2*(0.25*X**2 + gamma21bar))*Y**2 - A*(0.25*X**2 - gamma21bar**2)*Y + (1+phibar2)*(0.25*X**2 + gamma21bar)**2
exactPhase4 = (1 - phibar)*Y**4 + A*Y**3 + (1-phibar)*((1+gamma21bar)**2 - 2*(0.25*X**2 + gamma21bar))*Y**2 - A*(0.25*X**2 - gamma21bar**2)*Y + (1-phibar)*(0.25*X**2 + gamma21bar)**2

#PLOTTING----------------------------------------------------------------------
#Create Subplot
fig, ax = plt.subplots(1, 1, dpi=100)

# Plot numerical data points
# marker-line-color
ax.plot(OmegaCArrayNorm, SpecWidthMaxArray/gamma31, "ok", label="Data (Type 1)")
ax.plot(OmegaCArrayNorm, SpecWidthNMaxArray/gamma31, "^k", label="Data (Type 2)")
# ax.plot(OmegaCArrayNorm, SpecWidthAltMaxArray1/gamma31, "*k", label = "Data (Alt Peak 1)")
# ax.plot(OmegaCArrayNorm, SpecWidthAltMaxArray2/gamma31, "sk", label = "Data (Alt Peak 2)")

# plot quartic curve-fitting
# ax.plot(OmegaCArrayNorm, curve_fit1, "-r", label = "Quartic Curve-Fit (Type 1)")
# ax.plot(OmegaCArrayNorm, curve_fit2, "-b", label = "Quartic Curve-Fit (Type 2)")
# ax.plot(OmegaCArrayNorm, curve_fit3, "-g", label = "Quartic Curve-Fit (Alt PEak 1)")
# ax.plot(OmegaCArrayNorm, curve_fit4, "-m", label = "Quartic Curve-Fit (Alt Peak 2)")

# plot approximated, no additional phase shift term
# ax.plot(OmegaCArrayNorm, approxNoPhase1, "-c", label = "Approximated, Uncorrected (Type 1)")
# ax.plot(OmegaCArrayNorm, approxNoPhase2, "-y", label = "Approximated, Uncorrected (Type 2)")

# plot approximated, additional phase shift term
# ax.plot(OmegaCArrayNorm, approxPhase1, "-c", label = "Approximated, Corrected (Type 1)")
# ax.plot(OmegaCArrayNorm, approxPhase2, "-y", label = "Approximated, Corrected (Type 2)")

# plot approximated, additional phase term (implicit)
# CS1 = ax.contour(X, Y, approxPhaseImplicit1, [0], colors = "cyan")
# CS2 = ax.contour(X, Y, approxPhaseImplicit2, [0], colors = "yellow")
# ax.clabel(CS1, inline=1, fontsize=10)
# ax.clabel(CS2, inline=1, fontsize=10)
# labels = ["Approximated, Corrected (Type 1)", "Approximated, Corrected (Type 2)"]
# CS1.collections[0].set_label(labels[0])
# CS2.collections[0].set_label(labels[1])

# plot exact, no additional phase shift term
# CS1 = ax.contour(X, Y, exactNoPhase1, [0], colors = "cyan")
# CS2 = ax.contour(X, Y, exactNoPhase2, [0], colors = "yellow")
# ax.clabel(CS1, inline=1, fontsize=10)
# ax.clabel(CS2, inline=1, fontsize=10)
# labels = ["Exact, Uncorrected (Type 1)", "Exact, Uncorrected (Type 2)"]
# CS1.collections[0].set_label(labels[0])
# CS2.collections[0].set_label(labels[1])
    
#plot exact, additional phase shift term
CS1 = ax.contour(X, Y, exactPhase1, [0], colors="red", linestyles="solid")
CS2 = ax.contour(X, Y, exactPhase2, [0], colors="green", linestyles="dashed")
# ax.clabel(CS1, inline=1, fontsize=10)
# ax.clabel(CS2, inline=1, fontsize=10)
labels = ["Eq. (1) (Type 1)", "Eq. (2) (Type 2)"]
CS1.collections[0].set_label(labels[0])
CS2.collections[0].set_label(labels[1])

# level1 = CS1.collections[0]
# paths1 = level1.get_paths()
# del(paths[0])

# level2 = CS2.collections[0]
# paths2 = level2.get_paths()
# del(paths[3])

# for vertice in paths1[0].vertices:
#     if vertice[1] < 10:
#         None

ax.set_xlabel("$\Omega_{c}/\gamma_{31}$")
ax.set_ylabel("$\Delta_{p}/\gamma_{31}$")

ax.tick_params(axis="x")
ax.tick_params(axis="y")

# ax.legend(loc="upper left")
ax.grid(False)

# cs = ax.contourf(OmegaCGrid, detuningGrid, IsArray, cmap = "hot", vmin = 0, vmax = 4, levels = 100)
# colBar = fig.colorbar(cs)
# colBar.set_label(label="$I_{s}/I_{0}$", size=29)                
# colBar.ax.tick_params(labelsize = 29)

plt.tight_layout()
plt.show()

#END TIME-----------------------------------------------------------------
#sleeping for 1 sec to get 10 sec runtime
time.sleep(1)

#end time
end = time.time()

#TOTAL RUNTIME------------------------------------------------------------
#total time taken
print(f"Runtime of the program is {end - start}")