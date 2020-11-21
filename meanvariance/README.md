# Adaptation to stimulus mean and variance

Change into the `meanvariance/` directory and run
``` sh
python3 meanvariance.py
```
for a demo.

In the following, key concepts of adaptation to the mean, computation
of the amplitude modulation, and adaptation to the variance, as well
as the respective code are briefly described. See the
[`meanvariance.py`](meanvariance.py) script for the full functions.


## Adaptation to the mean

Let's generate a stimulus with stepwise different mean values. For
this we create a longer white-noise stimulus and a corresponding
array with the mean values that then is added to the noise stimulus.

``` py
dt = 0.001                    # integration time step in seconds
tmax = 4.0                    # stimulus duration in seconds
cutoff = 20.0                 # cutoff frequency of stimulus in Hertz
T = 1.0                       # duration of segements with constant mean in seconds
means = [0.0, 3.0, 6.0, 1.5]  # mean stimulus values for each segment
# white noise stimulus:
stimulus = 0.5*whitenoise(0.0, cutoff, dt, tmax)
time = np.arange(len(stimulus))*dt
# array with means:
mean = np.zeros(len(stimulus))
for k, m in enumerate(means):
    mean[(time>k*T) & (time<=(k+1)*T)] += m
# add mean values to stimulus:
stimulus += mean
# plot:
ax.plot(time, stimulus)
ax.plot(time, mean)
```

![meanstimulus](meanvariance-meanstimulus.png)

Compute the spike-frequency response of a non-adaptating neuron
(&#120572; = 0)
``` py
rate0, _ = adaptation(time, stimulus, alpha=0.0, taua=0.5)
```
and of a strongly adapting neuron (&#120572; = 0.2) to this stimulus
``` py
rate, _ = adaptation(time, stimulus, alpha=0.2, taua=0.5)
```
and plot the resulting spike frequencies:

![meanresponse](meanvariance-meanresponse.png)

Spike-frequency adaptation removes most of the different mean values,
but not completely. The stronger the adaptation (higher &#120572;),
the more the mean values will be attenuated. Because of the non-linear
shape of the neuron's *f-I* curves (spike frequencies cannot be
negative), parts of the stimulus below the mean are cut out. Whenever
the stimuls mean switches to higher values, the neuron transiently
responds with a high spike frequency that then decays to lower levels.
When the mean is switched to lower values, the neuron ceases firing
until it recovers from adaptation.


## Computing the amplitude modulation


## Adaptation to the variance

