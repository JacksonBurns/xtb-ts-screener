---
title: "`XTBTSScreener.jl` - Screening Likely Transition States with Julia and Machine Learning"
date: "March 2023"
author: "Burns, Jackson - MIT"
geometry: margin=2cm
---

# Abstract

In the chemical kinetics, quantum mechanical simulations are often used to probe the complexities of chemical reactions.
One particular use case is the prediction of rate constants and kinetic barriers for unknown reactions.
Experimental studies are time consuming, and with the many thousands of chemical reactions that are present in modern reacting systems computational investigation is key.
To do so, one must be able to predict the three-dimensional electronic structure of the starting materials, the products, and - most importantly - the transition state.
The former two requirements are relatively simple, and existing computational chemistry packages have few issues with performing this task.
Finding transition states, however, is a far more challenging step.

First, a 'guess' at the possible transition state must be created.
Historically, this has been done by hand with expert input and computationally inexpensive (but less accurate) simulation methods, though systems are now being developed to predict transition states directly.
Next, the possible transition state is subjected to an expensive quantum mechanics simulation, as mentioned earlier.
In this study and others like it, Density Functional Theory (DFT) is used to verify if the transition state is chemically valid (i.e. "converged").
This simulation technique scales by $$O(n^3)$$ for $$n$$ electrons in the system, thus simulation times can be weeks or longer.
Finally, this process must be repeated until the DFT simulation converges and a valid transition state is found.
In almost 50% of cases, the first suggested transition state will fail to converge, effectively wasting hundreds of compute hours.

![Current Transition State Search Workflow](https://github.com/JacksonBurns/1xtb-ts-screener/blob/main/paper/images/current_workflow_diagram.png?raw=true)

To accelerate this workflow it would be useful to be able to estimate _a-priori_ if a suggested transition state is likely to converge or not before moving it to the DFT simulation stage.
To do this, we have collected a dataset containing many examples of expert-suggested possible transition states for reactions of interest in the chemical kinetics field.
These structures were partially optimized using Extended Tight Binding semi-empirical quantum mechanics simulations (XTB), which is a computationally inexpensive method to arrive at a reasonable initialization.
All examples were then subjected to DFT simulation, and approximately half did not converge.
Using the files which were input to the DFT simulation, which contain the proposed transition state structure in the form of atomic coordinates, and the convergence or non-convergence of the DFT simulation we can train a machine learning (ML) model to classify suggested transition states as "like to converge" or not.
This is shown schematically in the diagram below, which is a modification of the current common workflow.

![Proposed Enhanced Transition State Search Workflow](https://github.com/JacksonBurns/1xtb-ts-screener/blob/main/paper/images/proposed_workflow_diagram.png?raw=true)

This ML model will be implemented using a Neural Network (NN) in Julia.
The Lux [@pal2022lux] package will be used to configure models and the Adam optimiser [@kingma2017adam] will be used in model training.
The dataset has graciously been provided for use by Oscar Wu of the Green Group.


# References