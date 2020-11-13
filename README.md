# Neural Adaptation Primer

## Requirements

The scripts run on `python` version 3, using the following packages:

- numpy
- scipy
- matplotlib


## LIFAC: leaky integrate-and-fire with adaptation current

Run
``` sh
python3 lifac.py
```
for a demo.

The leaky integrate-and-fire neuron is extended by an adaptation current *A*:

<img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+%5Cbegin%7Balign%2A%7D%0A%5Ctau_m+%5Cfrac%7BdV%7D%7Bdt%7D+%26%3D+-+V+%2B+RI+-+A+%2B+D%5Cxi+%5C%5C%0A%5Ctau_a+%5Cfrac%7BdA%7D%7Bdt%7D+%26%3D+-A%0A%5Cend%7Balign%2A%7D%0A" 
alt="\begin{align*}
\tau_m \frac{dV}{dt} &= - V + RI - A + D\xi \\
\tau_a \frac{dA}{dt} &= -A
\end{align*}
">

The leaky integration of the membrane potential *V(t)* with membrane
time constant *&#120591;<sub>m</sub>* is driven by a stimulus *RI* (input
resistance *R* times injected current *I(t)*) from which the
adaptation current *A* is subtracted. *D&#958;* is an additive white
noise. The adaptation current is integrated with the adaptation time
constant *&#120591;<sub>m</sub>*. Whenever the membrane voltage crosses the
firing threshold &#952;, a spike is generated, the adaptation current
is incremented by &#945;, the voltage is reset to *V<sub>r</sub>*, and
integration is paused for the absolute refractory period *&#120591;<sub>r</sub>*.

The `lifac()` function integrates the model using Euler forward integration:
``` py
def lifac(time, stimulus, taum=0.01, tref=0.003, noised=0.01,
          vreset=0.0, vthresh=1.0, taua=0.1, alpha=0.05, rng=np.random):
    dt = time[1] - time[0]                                # time step
    noise = rng.randn(len(stimulus))*noised/np.sqrt(dt)   # properly scaled noise term
    # initialization:
    tn = time[0]
    V = rng.rand()
    A = 0.0
    # integration:
    spikes = []
    for k in range(len(stimulus)):
        if time[k] < tn:
            continue                         # no integration during refractory period
        # membrane equation:
        V += (-V - A + stimulus[k] + noise[k])*dt/taum
        # adaptation dynamics:
        A += -A*dt/taua
        # threshold condition:
        if V > vthresh:
            V = vreset               # voltage reset
            A += alpha/taua          # adaptation increment
            tn = time[k] + tref      # refractory period
            spikes.append(time[k])   # store spike time
    return np.asarray(spikes)
```

Use this function by first defining a time vector and an appropriate stimulus:
``` py
dt = 0.0001               # integration time step in seconds
time = np.arange(0.0, 0.2+dt, dt)
stimulus = np.zeros(len(time))
stimulus[(time > 0.0) & (time < 0.1)] = 3.0
```
The time step `dt` sets the integration time step. Make sure that it is at least
ten times smaller than the membrane time constant.
Then call the `lifac()` function to simulate a single trial:
```
spikes, v, a = lifac(time, stimulus)
``` py
You then can plot the membrane voltage `v` and the adaptation current
`a` as a function of time `time`.

Or use the `multi_trials()` function to simulate several trials of the `lifac()` model
with the same stimulus:
``` py
spikes = multi_trials(20, time, stimulus, lifac)
```
The returned spikes are a list of arrays with spike times of each trial.

