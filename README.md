# Analytical Model of Alchemical Transfer
Simulation input files, analytical model of coupling and of transfer with corresponding data for the paper "Potential Distribution Theory of Alchemical Transfer" by Solmaz Azimi and Emilio Gallicchio. Pre-print can be accessed at https://arxiv.org/abs/2407.14713. 

The analytical model is initiated with a python script in each complex subdirectory in `optimization`. Mathematica notebooks for analysis of parameters are included for two processes: coupling and transfer. 

- `simulation-input`: input files of each calculation (hydration, coupling, and transfer) for each system based on AToM-OpenMM and OpenMM's ATMForce version 8.1.1
-  `data`: input data for the analytical model of each calculation (hydration, coupling, and transfer) for each system
-  `optimization`: sorted parameterization of coupling and hydration calculations, including driver scripts to initiate the analytical model
-  `scripts`: script to produce latex tables of results
