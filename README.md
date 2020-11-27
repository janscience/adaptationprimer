# Neural Adaptation Primer

This repository provides a tutorial on how to model neural
adaptation. It complements the primer on "Neural adaptation" by Jan
Benda published in Current Biology. Feel free to use and distribute
the scripts and figures for teaching. The code, figures, and
descriptions are provided under the [GNU General Public License
v3.0](LICENSE).

For each of the topic listed below, there is a folder 'TOPIC/' that
contains a 'README.md' file, a python script 'TOPIC.py' containing the
functions explained in the README.md file that you can run for a demo,
a python script 'TOPICplots.py' that uses the functions in 'TOPIC.py'
to generate the figures in the 'README.md' file, and these figures as
'TOPIC-*.png' files.

For running the demo script, change into the directory of the topic,
and run the script. For example, to run the demo for the adapting
leaky integrate-and-fire model in 'lifac/' do:
```
cd lifac
python3 lifac.py
```
Or simply open the 'lifac/lifac.py' script in your IDE and run it from
there.


## Requirements

The `python` scripts run in python version 3, using the following packages:

- numpy
- scipy >= 1.2.0
- matplotlib >= 2.2.0


## Leaky integrate-and-fire with adaptation current

The leaky integrate-and-fire model is a simple model of a spiking
neuron. Augmented with a generic adaptation current it reproduces many
features of intrinsically adapting neurons. [Read more in
`lifac/`.](lifac/README.md)


## Spike-frequency adaptation models

Spike-frequency adaptation is a phenomenon of the spike
frequency. Modeling adaptation on the level of spike freuqencies is
thus a natural choice. [Read more in `sfa/`.](sfa/README.md)


## Adaptation high-pass filter

Spike-frequency adaptation basically adds a high-pass filter to the
neuron's input-output function. This filter operation interacts with
the non-linear *f-I* curves of the neuron. [Read
more in `filter/`.](filter/README.md)


## Adaptation to stimulus mean and variance

Subtractive adaptation is perfectly suited to make the neuron's
response invariant with respect to the mean of the
stimulus. Invariance to the stimulus variance, however, requires
thresholding to extract the amplitude modulation and divisive
adaptation. [Read more in `meanvariance/`.](meanvariance/README.md)


## Stimulus-specific adaptation

Adaptation in parallel pathways leads to stimulus-specific
adaptation. [Read more in `ssa/`.](ssa/README.md)


## Resolving ambiguities

Absolute stimulus intensity is ambiguously encoded by an adapting
neuron. Nonetheless, matched intrinsic adaptation allows down-stream
neurons to robustly encode absolute stimulus intensity. [Read more in
`ambiguities/`.](ambiguities/README.md)


## Generating sparse codes

Efficient codes are both temporally and spatially sparse. Intrinsic
adaptation together with lateral inhibition generate such sparse
codes. [Read more in `sparse/`.](sparse/README.md)


## Contributing

You are welcome to improve the code and the explanations. Or even add
another chapter.

Fork the repository, work on your suggestions and make a pull request.

For minor issues, e.g. a reference you want me to add, or fixing
little quirks, open an issue.
