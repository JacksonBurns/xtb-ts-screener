---
title: "`XTBTSScreener.jl` - Screening Likely Transition States with Julia and Machine Learning"
date: "March 2023"
author: "Jackson Warner Burns _Computational Science and Engineering, MIT_"
geometry: margin=2cm
---

# Abstract

In chemical kinetics, quantum mechanical simulations are often used to probe the complexities of novel reactions.
One particular use case is the calculation of rate constants.
This requires predicting the three-dimensional electronic structure of the starting materials, the products, and - most importantly - the transition state.
The former two requirements are relatively simple and existing computational chemistry packages readily perform this task.
Finding transition states is substantially more difficult.
The typical workflow for doing so is shown in Figure 1 and explained in-depth below.

![Current Transition State Search Workflow](https://raw.githubusercontent.com/JacksonBurns/xtb-ts-screener/main/paper/images/current_workflow_diagram.png){ height=125px }

First, a 'guess' at the possible transition state is created.
Creating this initialization has historically been done by hand with expert input and computationally inexpensive but less accurate simulation methods, though systems are now being developed to predict transition states directly [@tsguessgan].
Next, the possible transition state is subjected to an expensive quantum mechanics simulation.
In this study, Density Functional Theory (DFT) is used to verify if the transition state is chemically valid.
Validity is indicated by the DFT simulation finishing normally (i.e. the simulation "converged") rather than terminating mid-execution.
DFT scales by $O(n^3)$ for $n$ electrons in the system, thus simulation times can be weeks or longer for common systems of interest.
If not converged, this entire process must be repeated until the DFT simulation is successful and a valid transition state is found.
Every failed calculation effectively wastes hundreds of compute hours.

To accelerate this workflow it would be useful to estimate _a-priori_ if a suggested transition state is _"likely to converge"_ before simulating with DFT.
To study this, members of the Green Group have collected a dataset containing many thousands of expert-suggested possible transition states for reactions of interest in the chemical kinetics field.
These structures were partially optimized using Extended Tight Binding semi-empirical quantum mechanics simulations (xTB), which is a computationally inexpensive method to arrive at a reasonable initialization.
All examples were then carried forward to DFT and approximately 50% did not converge, i.e. failed to produce a valid transition state from the initialization.
By using the atomic coordinates of these proposed transition state structures as features and the convergence or non-convergence of the corresponding DFT simulation as a label, we can train a machine learning (ML) model to classify suggested transition states as _"likely to converge"_ or not: `XTBTSScreener.jl`.
This is shown schematically in Figure 2.

![Proposed Enhanced Transition State Search Workflow](https://raw.githubusercontent.com/JacksonBurns/xtb-ts-screener/main/paper/images/proposed_workflow_diagram.png){ height=125px }

This ML model will be implemented using a Neural Network (NN) in Julia.
The longer-term use case of `XTBTSScreener.jl` would be within a much larger closed-loop optimization tool, so the speed of Julia will be critical in making this approach worthwhile.
The `Lux.jl` [@pal2022lux] package will be used to configure models and the Adam optimiser [@kingma2017adam] will be used in model training to enable more rapid convergence.
`Zygote.jl` [@Zygote.jl-2018] is also incorporated to provide automatic differentiation capabilities for the NN.
If initial modeling efforts are unsuccessful, a Graph Neural Network (GNN) will be used via `GraphNeuralNetworks.jl` [@Lucibello2021GNN].
Literature precedent from the chemical informatics field at large indicates that GNNs often perform better than typical NNs on chemical data.
The dataset of proposed transition states partially optimized by xTB and their corresponding converged/failed label has been graciously provided for use by Haoyang Wu of the Green Group.
The authors acknowledge the MIT SuperCloud and Lincoln Laboratory Supercomputing Center for providing HPC resources that have contributed to the generation of this dataset [@reuther2018interactive].


# References