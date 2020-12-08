README for Neural nets for bioprocesses 2020

Author: Adam Coxson, MPhys Undergrad, The University of Manchester
Email: adamcoxson1@gmail.com
Project: Evaluating hyperparameter optimisation techinques for an FNN and RNN, Aug 2020
Main report: Comparing FNN and RNN performance (Polished): https://www.overleaf.com/read/xctrqdqybckj
Secondary report: Comparing hyperparameter optimisations method (very unpolished): https://www.overleaf.com/read/hdmdcfkqyrpx
Google drive for entire project: https://drive.google.com/drive/folders/14E8SYFzoncRtdEVwSguZaxXrR-hf7ufy?usp=sharing

This was for my SEI internship project into hyperparameter optimisation techniques
and neural networks for bio-processes. My supervisor was Dr Dongda Zhang from the 
School of Chemical Engineering and Analytical Sciences, Uni of Manchester.
Email: dongda.zhang@manchester.ac.uk

The main aim of my project was to compare the performance of FNNs and RNNs in offline data prediction for
bio-process data. This was completed in the final 2 weeks of my 8 week project. The Secondary aim of my
project was to compare different hyperparameter optimisation techniques which took up the first 6 weeks.
A lot of the initial development code was for my secondary project aim, and is quite messy, see folder info below.
SPOILER: Bayesian Optimisation was the best hyperparameter optimisation technique, I made modifications to
the testing and training procedure to obtain test error means and standard deviations to use in the 
Bayesian search. This made the Bayesian Optimisation extremely effective. I then used this to form optimised
neural networks for the final comparison between the FNN and RNN. See slides 75-77 of the "Weekly journal" 
powerpoint file to see what I mean, or refer to the reports.

The 5 folders in this directory contain all the code files, documentation and resources I used.

Hyperparameter optimisation codes:
This folder contains developmental code, data files and documentation. I have not bothered to organise any of it.
Some information may be gleaned from it. The most significant thing will be implementations of the Evolutionary 
algorithms which I investigated and ruled out quite early on during my project. Note, the testing and training 
procedure in the codes found here are sub-optimal and can have high variance in performance between runs.
Admittedly, I would struggle to sort through it at points. You have been warned, don't judge me!

Neural Network code (FNN and RNN):
This folder contains polished code for handover purposes. I have fully commented the code where relevant. 
The code has been generalised to a significant degree although there is room for plenty more. This code should
work out of the box. 
See the README within for more details.

Papers and resources:
Just pdfs of papers, textbooks and useful links. Somewhat organised.

Weekly slides and report:
This contains miscellaneous and powerpoint files which I used to document my weekly progress to show to Dongda 
during our weekly monday meetings. The ppt which has all of this is the "Weekly journal" ppt file. The folder also 
contains some fragmented files from my report planning. Just use the overleaf links provided above to find my
reports and download the pdfs.

Code Mostafizor Rahman UoM:
This file contains all the code from M. Rahman who did a similar case study for his MEng project. My code was
initially based off of his. Email Dongda for more details.


