## Thesis Project

A Thesis in the Field of Mathematics and Computation

Title: Modeling Heterogeneous Seasonality with Recurrent Neural Networks Using IoT Time Series Data for Defrost Detection and Anomaly Analysis


LIST OF FILES:  
* Thesis_Suraj_Khetarpal_Final.pdf:  This is the thesis document.
* Defrost_Detection_with_RNNs.ipynb: This notebook contains the code for building and testing the LSTM networks, and for generating all figures.
* Refrigeration_Unit_Simulation.py:  This file contains code for simulating refrigeration temperature time series data.
* Generate_Binary_Datasets.py:  This file contains code for generating the binary datasets that were used to test the ability of LSTM networks to model heterogeneous periodicity.


ABSTRACT:  
Detecting anomalies and predicting failure of industrial refrigeration equipment are paramount to guarantee a reliable supply chain and consumer safety in a variety of industries such as pharmaceutical and grocery. Failure can be predicted by performing an anomaly analysis on internal temperature data that has been collected by Internet of Things (IoT) sensors. Such an analysis involves monitoring a refrigeration unit’s defrost cycle, which is the seasonal (i.e. periodic) component of its temperature time series.
In many industries, Recurrent Neural Networks (RNNs) are used to analyze and forecast time series data, and Long Short Term Memory (LSTM) cells are used to remember long term dependencies. When using deep learning tools to analyze time series data, a major challenge is modeling datasets with heterogeneous seasonal components.
This thesis investigates the ability of RNNs built with LSTM cells to detect defrost events from within temperature time series that were recorded using IoT sensors. Because defrost events are seasonal and heterogeneous across time series, the successful detection of defrosts is dependent on an RNN’s ability to build a model of heterogeneous seasonality.
We conducted our research in two phases. During the first phase, we generated datasets of simulated refrigeration temperature time series and used them to train and test RNNs, resulting in a 95% classification accuracy rate. During the second phase, we analyzed the challenges inherent in modeling heterogeneous seasonality in our time series datasets. We designed new experiments and modeled heterogeneous seasonality using binary datasets consisting of 1s and 0s. Using a binary dataset whose seasonal patterns were heterogeneous in both frequency and phase, RNNs achieved 100% forecasting accuracy. However, when we added confounding features to the dataset, forecasting accuracy dropped to 71% as confounding features could easily be confused with the seasonal patterns of time series data. Our results revealed that RNNs are best suited to model heterogeneous seasonality in datasets that do not contain confounding features.
