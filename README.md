[![Python Windows Tests](https://github.com/jamadaha/P5/actions/workflows/windowsTests.yml/badge.svg?branch=main&event=push)](https://github.com/jamadaha/P5/actions/workflows/windowsTests.yml)
[![Python Linux Tests](https://github.com/jamadaha/P5/actions/workflows/linuxTests.yml/badge.svg?branch=main&event=push)](https://github.com/jamadaha/P5/actions/workflows/linuxTests.yml)

# JANGAN
### Introduction
This is a semester project for a group of students from Aalborg University (AAU).
This semester was about working with either Machine Learning (ML) and/or Databases (DB), where this group choose to go with ML.
The particular problem that this group have tryed to solve was that of data imbalance when it comes to training ML models. This is a significant issue withing ML, since models become significantly worse when training on unbalanced data, to a degree where the output they are trying to make, simply is just noise.
So this project have attempted, by means of experimentation, to implement a CGAN and a Classifier in tandem. The goal is for the CGAN to be able to take in unbalanced data and then generate a balanced dataset that the Classifier can then determine to be "correct" or not.
The dataset used is the [MNIST](https://www.nist.gov/srd/nist-special-database-19) special dataset, that consist of handwritten letters.

### How to use
Most of the JANGAN is controlled by means of Config Files, where there are two major variants. The `ExperimentQueueConfig.ini` and the experiment configs (example being `BaseRunConfig.ini`)
Desciptions of how each config works are in the config files themselfs. For this repo, there is also a lot of configs for experiments, that can be run if interested.
To make a new experiment, all you need to do is add a new config in the `/ExperimentQueue/*.ini` folder and a `*.py` file to accompany it. Whatever you now may want to do in the experiment can be done in that `*.py`. Then by adding a new category for the experiment to the `ExperimentQueueConfig.ini`, just like the rest of them are, and setting it to run in the `ExperimentList` list, the experiment will run.

Then to run it, in a console type `python3 JANGANQueue.py` to start the queue. While the JANGAN is running, the console output is filled with debugging information, that can be used for experiments.

### Output
The JANGAN does a lot of different logging both for the CGAN and for the Classifier. All these are put in the folder that is defined in the start of the experiment config (being the `BasePath` key). The classifier makes some .csv files for how the training and classification went, however it also makes a Confusion Matrix in the logging folders, that gives a good overview of how well the CGAN output was.
<img src="https://user-images.githubusercontent.com/22596587/146333725-b0498637-b1c0-406e-a8cb-279aa861dd8e.png" width="300" />

As well as the CGAN makes a path full of new output images, that can be viewed in the logging folder. It also creates an image for each epoch it goes through, where a progression of how well each class label is doing can be seen.

<p>
<img src="https://user-images.githubusercontent.com/22596587/146334376-779ae688-a524-427f-853c-b91f3552e95c.png" width="300" />
  =>
<img src="https://user-images.githubusercontent.com/22596587/146334241-211fea6d-9d86-407c-bd8e-d00dc76ae766.png" width="300" />
</p>

