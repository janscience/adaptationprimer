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
this we generate a longer white-noise stimulus and a corresponding
array with the mean values that then is added to the noise stimulus.

``` py
dt = 0.001              # integration time step in seconds
tmax = 4.0              # stimulus duration in seconds
cutoff = 50.0           # cutoff frequency of stimulus in Hertz
stimulus = 0.5*whitenoise(0.0, cutoff, dt, tmax)
time = np.arange(len(stimulus))*dt
mean = np.zeros(len(stimulus))
mean[(time>1.0) & (time<=2.0)] += 3.0
mean[(time>2.0) & (time<=3.0)] += 6.0
mean[(time>3.0) & (time<=4.0)] += 1.5
stimulus += mean
ax.plot(time, stimulus)
ax.plot(time, mean)
```

![meanstimulus](meanvariance-meanstimulus.png)


## Computing the amplitude modulation


## Adaptation to the variance

