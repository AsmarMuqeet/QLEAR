# QLEAR

1  Basic Installation

The codebase is only tested on Windows 11. For basic installation, the following softwares are required

- Python Anaconda distribution
- R distribution for Statistical Tests
- Matlab R2023a for Qraft method
- IBM API-Key for executing jobs on real quantum computers

Step 1. Cd to submission directory

Step 2. Open the Anaconda terminal and create a new virtual environment by running conda env create -f environment.yml

Step 3. Activate virtual environment by running conda activate BoF Step 4. Paste IBM API key in file API~~ KEY.txt

2  Data Files

All the data files required for reproducibility are included in the code folder.

- Directory--training_data/ contains the generated training data for ML models.
- Directory--testing_data/ contains the generated test data generated using the simulator for each computer.
- Directory--real_circuits/ contains the application-level circuit simulated executions for each computer.
- Directory--real_circuits_hardware/ contains the application-level circuit real executions for each computer
- Directory--real_circuits_qraft/ contains the application-level circuit simulated executions using Qraft feature calculations for each computer.
- Directory--real_circuits_hardware_qraft/ contains the application-level circuit real executions using Qraft feature calculations for each computer.
- File--outputerror_application_level_hardware.csv Contains Summerized data for RQ2 Table 4(a) real computers.
- File--outputerror_application_level_simulator.csv Contains Summerized data for RQ2 Table 4(a) simulator.
- File--outputerror_computer_level_hardware.csv Contains Summerized data for RQ2 Table 4(b) real computers.
- File--outputerror_computer_level_simulator.csv Contains Summerized data for RQ2 Table 4(b) simulators.
- File--statistical_test.xlsx Contains the full Mann-Whitney U test data for RQ3.
- File--A12_test.xlsx Contains the full A12 effectsize magnitude data for RQ3.
3  Evaluate RQs

Evaluation of RQ2 and RQ3 is done in the following Jupyter Notebook files

- File–RQ2 Hardware.ipynb Contains all the code to evaluate ML models on Real computer data. Rerun all the cells to produce results for RQ2 for real Computers
- File–RQ2 Simulator.ipynb Contains all the code to evaluate ML models on Simulator data. Rerun all the cells to produce results for RQ2 for Simulators.
- File–RQ3.ipynb Contains all the code for boxplot and statistical tests for RQ3. Rerun all the cells to produce results for RQ3.
4  Re-Evaluate Whole Experiment

NOTE: Execution time for running the whole experiment depends on the specifications of the machine and also on the waiting times for IBM quantum Computers

Step 1. Generate New Data by running the following files:

(Generate Training Data Simulator.ipynb, Generate Test Data Simulator.ipynb, Generate Test Data Hardware.ipynb, Generate Test Data-QRAFT.ipynb,

Generate Test Data-QRAFT-hardware.ipynb)

Step 2. Train QRAFT model

- cd QRAFT
- cd code_train
- Run train.m in matlab
- Copy prediction.m and QRAFT.mat to submission folder
- In submission folder run command python Qraft_Prediction.py and save Qraft predictions

Step 3. Trian ML models by running file Train ML models.ipynb

Step 4. Prepare Data for RQ3 by running command

python MLP10.py

python LOCO.py

Step 5. Repeat ALL steps in the section (Evaluate RQs). All Data files are now

updated.