---
title: "`XTBTSScreener.jl` - Saving CPU Cycles with Julia and Machine Learning"
date: "April 2023"
author: "Jackson Warner Burns _Computational Science and Engineering, MIT_"
geometry: margin=2cm
---

# Abstract
In chemical kinetics, proposed chemical reaction transition states are optimized with computationally expensive simulations for subsequent analysis and investigation.
Often these simulations will fail to converge and require multiple iterations to achieve the desired result, which wastes compute hours and impedes research.
Using Julia and Machine Learning `XTBTSScreener.jl` is created as a proof-of-concept to predict which proposed transition states are likely to converge _before_ investing compute hours.
`XTBTSScreener.jl` uses the output from inexpensive semi-empirical simulations, which can be executed in only a few minutes on consumer-grade hardware, to predict if the partially-optimized structure is likely to converge.
The results of this initial investigation are promising and indicate that a combination of the atomic coordinates and higher-order electronic descriptors can enable a Long Short-Term Memory RNN [@lstmnetworks]  to achieve better-than-baseline performance on this task.

# Background
In chemical kinetics, quantum mechanical simulations are often used to probe the complexities of novel reactions.
One particular use case is the calculation of rate constants [@Spiekermann2022].
This requires predicting the three-dimensional electronic structure of the starting materials, the products, and - most importantly - the transition state between the two.
The former two requirements are relatively simple and existing computational chemistry packages readily perform this task.
Finding transition states is substantially more difficult.
The typical workflow for doing so is shown in Figure 1 and explained in-depth below.

![Current Transition State Search Workflow](https://raw.githubusercontent.com/JacksonBurns/xtb-ts-screener/main/paper/images/current_workflow_diagram.png){ height=125px }

First, a 'guess' at the possible transition state is created.
Creating this initialization has historically been done by hand with expert input and computationally inexpensive, but less accurate, simulation methods.
One class of such methods which are used in this investigation are the "semi-empirical" methods which use shortcuts in solving the Hamiltonian based on experimental observations and human intuition.
Systems are now being developed to predict transition states directly [@tsguessgan], but they are imperfect.
Next the possible transition state is subjected to an expensive quantum mechanics simulation.
In this study, Density Functional Theory (DFT) is used to optimize the _proposed_ transition state to hopefully arrive at a _valid_ transition state.

Validity is determined by subsequent steps in the workflow which are beyond the scope of this study, as they involve applying further quantum mechanics simulations.
This work instead focuses on predicting if the DFT simulation will _finish execution normally_ (i.e. the simulation "converged") rather than terminating mid-execution, which is a risk inherent in the process of proposing transition states.
Termination has a variety of different and hard-to-detect causes.
If a proposed transition state is trapped in a local minimum on the energy surface or on an optimization trajectory that is not productive, DFT simulations will fail to converge.

This challenge in and of itself would not be worth addressing if it were not for the cost of failed simulations.
DFT scales by $O(n^3)$ for $n$ electrons in the system, thus simulation times can be weeks or longer for common systems of interest.
If not converged, this entire process must be repeated until the DFT simulation.
Every failed calculation effectively wastes hundreds of compute hours and dramatically slows research progress.

To accelerate this workflow it would be useful to estimate _a-priori_ if a proposed transition state is _"likely to converge"_ before simulating with DFT. 
By generating a dataset of proposed transition states and their failure or convergence label, it is possible to train a machine learning model that is capable of making said prediction: `XTBTSScreener.jl`.
This is shown schematically in Figure 2.

![Proposed Enhanced Transition State Search Workflow](https://raw.githubusercontent.com/JacksonBurns/xtb-ts-screener/main/paper/images/proposed_workflow_diagram.png){ height=125px }

This diagram adds a new step before proceeding to DFT simulation in which proposed transition states that are _immediately_ thought to be unlikely to converge will be rejected.
The method by which the proposed transition state was generated would then be responsible for creating a new, better-informed guess before continuing in the loop.
In doing so, failed DFT simulations are avoided and the loop operates faster.
Owing to the speed of Julia this screening step can be performed quickly so as not to defeat the purpose of rapid transition state proposal and simulation.

# Data Preparation
Members of the Green Group have previously collected a dataset containing many thousands of expert-suggested proposed  transition states for reactions of interest in the chemical kinetics field.
These structures were partially optimized using Extended Tight Binding semi-empirical quantum mechanics simulations (xTB), which is a computationally inexpensive method to arrive at a reasonable initialization [@xtbreview].
All examples were then carried forward to DFT simulation, resulting in both converged and failed simulations.
Currently, the data resides in log files output from the quantum chemistry simulation software that ran the simulations.
Data were retrieved synchronously with this study but the precise implementation details are beyond the scope of investigation.

To begin, approximately 20,000 simulation results were parsed and exported into first a SQL database and then a CSV file for easy loading into Julia.
Initial investigation revealed that the data were unbalanced (approximately 80% converged), so for this proof of concept the data were further downsampled to 7,000 samples with an even mix of failed and converged simulations.
At the scale of the entire dataset with _millions_ of samples, the balance between failed and converged simulations is much closer.

The data actually retrieved from the log files were the standardized atomic coordinates of the proposed transition state structures and the Gibbs free energy, $E_{0}+ZPE$, and number of simulation steps for each of the three sub-steps in the xTB simulation.
Atomic coordinates reflect the actual orientation of the atoms in space and embed chemical data like bond length and angles.
The Gibbs free energy and $E_{0}+ZPE$ (sum of electronic total energy and zero point energy) are two higher-order electronic descriptors that inform the 'absolute' energy of the transition state.
Number of simulation steps is included as a catch-all metadeta descriptor for the xTB simulations describing at a high level the 'difficulty' of the simulation.
Finally, the label for each point is simply whether or not the corresponding DFT simulation converged.

# Methods
`XTBTSScreener.jl` is implemented using Julia v1.8.5.
The grander vision for `XTBTSScreener.jl` and its derivatives would be within a much larger closed-loop optimization tool, so the speed of Julia will be critical in making this approach worthwhile.
`Lux.jl` [@pal2022lux], a low-code explicit parameterization-driven neural network package, is used to implement the model.
Sixty input dimensions and six hidden dimensions and the sigmoid activation function were used alongside the binary cross-entropy function to quantify loss, which is limited in numerical stability but offers benefits in interpretability.
The model itself is a Long Short-Term Memory recurrent neural network [@lstmnetworks], which is able to process higher-dimensional inputs like those in this study.
The ubiquitous ADAM optimizer [@kingma2017adam] is used in model training to enable more rapid convergence.
`Zygote.jl` [@Zygote.jl-2018] is also used to provide automatic differentiation capabilities for the network.

To parameterize the transition states, the aforementioned parameters are concatenated into a single input array.
Owing to the nature of chemical systems, the transition states in this dataset range in size from thirty to fifty atoms.
To maintain a uniform encoding and preserve interpretability of feature arras, zero-padding is used.
As later explained in the Future Work section, this simple encoding strategy should likely be replaced in the future with a more robust approach, but as a proof of concept it is sufficient.

# Results
There is no established best practice for training a LSTM on this type of data, so parameters like the batch size, learning rate, and number of epochs had to be determined by trial and error.
To establish a baseline model performance against which subsequent efforts could be compared, a network was trained using an "off the shelf" configuration.
Only the atomic coordinates are provided as features, the data was _not_ downsampled, and the NN hyperparameters were left to their defaults as specified in the `Lux.jl` tutorial [@pal2022lux]: learning rate of 0.01, batch size of 128, and 25 epochs.
This produced the loss and accuracy plot show below.

![Baseline Modeling Results](https://raw.githubusercontent.com/JacksonBurns/xtb-ts-screener/main/src/result-2023-04-23T12%3A30%3A43.862-fullrunfixedlegend.png){ height=150px }

Learning rate was varied from 0.01 to 0.0001, the latter of which was empirically determined to be critical for model convergence. 
Batch sizes between 16 and 128 samples, with 64 found to be the ideal value for avoiding over-fitting while still enabling convergence in a reasonable number of epochs.
Number of epochs was allowed to go as high as many thousands and as low as only ten, but with the above combination or parameters the loss function leveled out after only fifty epochs.
Following the example by Avik Pal in the Lux documentation, a Long Short Term Memory RNN was trained on the data.
All coordinate matrices were zero-padded to a uniform length for ease of encoding, with 55 input dimensions and 6 hidden dimensions and a sigmoid activation.
The ADAM optimiser was used with a learning rate of 0.01 and the binarycrossentropy was used for measuring loss.
This initial modeling was unsuccessful, producing this loss/accuracy curve:


Accuracy decreases with increasing epochs despite a decrease in the loss function, indicating model parameters were not conducive to learning or the embedding is masking information.
To attempt to improve performance, the learning rate was reduced to 0.0001 and the batch size was set to 2^4 from 2^6.

The fact that the learning rate has minimal impact on the results indicates the embedding used initially is not informative.
This is not entirely surprising, as this baseline model primarily is used as a baseline.
Most modern approaches require the addition of more simulation parameters to get results.

To do so, we will add the gibbs energy of each of the three optimization steps which shows change over time, the e0 zpe for each, and the number of steps required in each optimization to reach convergence (for the semiempirical simulations).


With these additional descriptors, the model performance stays almost flat across the epochs.
This indicates that the embedding is more meaningful but we might now be data limited.
Additional data were retrieved from the dataset.

Started by increasing the number of epochs.
Then further reduced learning rate.

# Conclusions

# Future Work
The dataset is imbalanced and should be expanded with more negative examples.
Additional descriptors could be added.
Graph networks could be added as well.
Zero padding is one of the simplest approaches but not the most rigorous, could be replaced with other encodings or some variety of autoencoding scheme.
If initial modeling efforts are unsuccessful, a Graph Neural Network (GNN) will be used via `GraphNeuralNetworks.jl` [@Lucibello2021GNN].
Literature precedent from the chemical informatics field at large indicates that GNNs often perform better than typical NNs on chemical data.

# Acknowledgements
The author thanks Green Group member Haoyang Wu for performing the calculations and providing the data which was used in this study.

The author acknowledges the MIT SuperCloud and Lincoln Laboratory Supercomputing Center for providing HPC resources that have contributed to the generation of this dataset and training of `XTBTSScreener.jl` [@reuther2018interactive].


# References