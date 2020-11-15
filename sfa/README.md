# Spike-frequency adaptation models

Run
``` sh
python3 sfa.py
```
for a demo.

In the following key concepts of the model and the respective code are
briefly described. See the `sfa.py` script for the full functions.


## Subtractive adaptation

Spike-frequency adaptation is caused by many different kinds of
adaptation current. For example, voltage-gated M-type current or
calcium activated potassium currents. Since these currents flow across
the cell membrane in parallel to all other ionic currents, inclusively
the input current, they subtractively act on the input current. Their
dynamics usually is a first order low-pass filter driven by the
spike-frequency of the neuron. This results in the following model
(see Benda and Herz, 2003, for details):

<img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+%5Cbegin%7Balign%2A%7D%0Af+%26%3D+f_0%28I-A%29+%5C%5C%0A%5Ctau_a+%5Cdot+A+%26%3D+-+A+%2B+%5Calpha+f%0A%5Cend%7Balign%2A%7D%0A" 
alt="\begin{align*}
f &= f_0(I-A) \\
\tau_a \dot A &= - A + \alpha f
\end{align*}
">

where *&#120591;<sub>a</sub>* is the adaptation time constant, &#945;
is the adaptation strength, *f(t)$ is the spike frequency,
*f<sub>0</sub>(I)* is the onset (unadapted) *f-I* curve, *I(t)* the
input, and *A(t)* is the adaptation level, i.e. the temporally
averaged adaptation current.

Let's model a saturating onset *f-I* curve by the upper half of a
Boltzman function:

<img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+%5Cbegin%7Balign%2A%7D%0Af_0%28I%29+%26%3D+%5Cleft%5C%7B+%5Cbegin%7Barray%7D%7Brcl%7D+%5Cfrac%7B2%7D%7B1%2Be%5E%7B-k%2A%28I-I_0%29%7D%7D+-+1+%26+%3B+%26+I+%3E+I_0+%5C%5C+0+%26+%3B+%26+I+%3C+I_0+%5Cend%7Barray%7D+%5Cright.%0A%5Cend%7Balign%2A%7D%0A" 
alt="\begin{align*}
f_0(I) &= \left\{ \begin{array}{rcl} \frac{2}{1+e^{-k*(I-I_0)}} - 1 & ; & I > I_0 \\ 0 & ; & I < I_0 \end{array} \right.
\end{align*}
">

With that we can implement the model using the Euler forward method:
``` py
def adaptation_sigmoid(time, stimulus, taua=0.1, alpha=1.0, taum=0.01, slope=4.0, I0=0.0):
    # sigmoidal onset f-I curve:
    f0 = lambda I: 2.0/(1.0+np.exp(-slope*(I-I0))) - 1.0 if I > I0 else 0.0
    dt = time[1] - time[0]           # integration time step
    # initialization:
    output = np.zeros(len(stimulus))
    adapt = np.zeros(len(stimulus))
    a = 0.0
    f = 0.0
    for k in range(int(3*taua//dt)):
        a += (alpha*f - a)*dt/taua
        f = f0(stimulus[0] - a)
    for k in range(len(stimulus)):
        a += (alpha*f - a)*dt/taua
        if taum < 2*dt:
            f = f0(stimulus[k] - a)
        else:
            f += (f0(stimulus[k] - a) - f)*dt/taum
        adapt[k] = a
        output[k] = f
    return output, adapt
```


## References

> Benda J, Herz AVM (2003) A universal model for spike-frequency adaptation. *Neural Comput.* 15, 2523-2564.

> Benda J, Hennig RM (2008) Dynamics of intensity invariance in a primary auditory interneuron. *J Comput Neurosci* 24: 113-136.

> Benda J, Longtin A, Maler L (2005) Spike-frequency adaptation separates transient communication signals from background oscillations. *J Neurosci* 25: 2312-2321.
