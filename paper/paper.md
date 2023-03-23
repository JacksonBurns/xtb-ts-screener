---
title: "`XTBTSScreener.jl` - Screening Likely Transition States with Julia and Machine Learning"
date: "March 2023"
author: "Jackson Warner Burns, _Computational Science and Engineering MIT_"
geometry: margin=2cm
---

# Abstract

In the chemical kinetics, quantum mechanical simulations are often used to probe the complexities of chemical reactions.
One particular use case is the prediction of rate constants and kinetic barriers for unknown reactions.
Experimental studies are time consuming, and with the many thousands of chemical reactions that are present in modern reacting systems computational investigation is key.
To do so, one must be able to predict the three-dimensional electronic structure of the starting materials, the products, and - most importantly - the transition state.
The former two requirements are relatively simple, and existing computational chemistry packages have few issues with performing this task.
Finding transition states, however, is a far more challenging step.
The typical workflow for doing so is shown in the Figure 1 and explained in-depth below.

![Current Transition State Search Workflow](https://raw.githubusercontent.com/JacksonBurns/xtb-ts-screener/main/paper/images/current_workflow_diagram.png){ height=125px }

First, a 'guess' at the possible transition state must be created.
Historically, this has been done by hand with expert input and computationally inexpensive (but less accurate) simulation methods, though systems are now being developed to predict transition states directly.
Next, the possible transition state is subjected to an expensive quantum mechanics simulation, as mentioned earlier.
In this study and others like it, Density Functional Theory (DFT) is used to verify if the transition state is chemically valid.
Validity is indicated by the DFT simulation finishing normally (i.e. the simulation "converged") rather than terminating mid-execution.
DFT scales by $O(n^3)$ for $n$ electrons in the system, thus simulation times can be weeks or longer for the systems of interest in this study.
If not converged, this entire process must be repeated until the DFT simulation is successful and a valid transition state is found.
When the suggested transition state fails to converge, the process is repeated hundreds of compute hours are effectively wasted.

To accelerate this workflow it would be useful to be able to estimate _a-priori_ if a suggested transition state is likely to converge before simulating with DFT.
To do this, we have collected a dataset containing many examples of expert-suggested possible transition states for reactions of interest in the chemical kinetics field.
These structures were partially optimized using Extended Tight Binding semi-empirical quantum mechanics simulations (xTB), which is a computationally inexpensive method to arrive at a reasonable initialization.
All examples were then carried forward to DFT and approximately 50% did not converge, failing to produce a valid transition state.
Using the proposed transition state structure in the form of atomic coordinates and the convergence or non-convergence of the DFT simulation we can train a machine learning (ML) model to classify suggested transition states as "like to converge" or not.
This is shown schematically in Figure 2, which is a modification of the current "common" workflow.

![Proposed Enhanced Transition State Search Workflow](https://raw.githubusercontent.com/JacksonBurns/xtb-ts-screener/main/paper/images/proposed_workflow_diagram.png){ height=125px }

This ML model will be implemented using a Neural Network (NN) in Julia.
The long-term positioning of the diagram shown above would be within a much larger closed-loop optimization tool, so the speed of Julia will be critical in making this approach worthwhile.
The `Lux.jl` [@pal2022lux] package will be used to configure models and the Adam optimiser [@kingma2017adam] will be used in model training to enable more rapid convergence.
`Zygote.jl` [@Zygote.jl-2018] is also incorporated to provide automatic differentiation capabilities for the NN.
If initial modeling efforts are unsuccessful, a Graph Neural Network will be used via `GraphNeuralNetworks.jl` [@Lucibello2021GNN] since literature precedent indicates that they often perform better than typical neural networks on chemical data.
The dataset of proposed transition states partially optimized by DFT and their corresponding converged/failed label has graciously been provided for use by Haoyang Wu of the Green Group.
The authors acknowledge the MIT SuperCloud and Lincoln Laboratory Supercomputing Center for providing HPC resources that have contributed to the generation of this dataset [@reuther2018interactive].


# References