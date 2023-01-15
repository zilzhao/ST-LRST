# Code for ST-LRST
- Folder 'Ano_syn_#1', 'Ano_syn_#2', and 'Ano_syn_#3' are the data of single type anomalies with different scales. Folder 'Ano_syn_mix' is the data for mixed type anomalies. Each folder contains:
  - Spatial_anomaly_infor_[#1/#2/#3/mix].csv contain anomaly information of simulation anomalies at different scales, where
       day: anomaly date, point_x: anomaly road ID, point_y: anomaly time period, impact_x: impacted road ID (based on topological adjacency),        impact_y: impacted time scope, degree: anomaly intensity.
  - Spatial_anomaly_mat_[#1/#2/#3/mix].csv are true values of simulated anomalies.
  - Spatial_ave_mat.csv is traffic speed under normal conditions (row: road segment, column: time interval)
  - Spatial_syn_mat_[#1/#2/#3/mix].csv are traffic speed after injecting traffic anomalies (#1/#2/#3: single-type, mix: mixed-type).
  - *ST-LRST.py is the code of the anomaly detection algorithm proposed in this paper.*

- AA_xian.csv is the adjacency matrix A of the urban road network.
- Free_speed.csv represents free flow speed of each road.

# Data source: 
DIDI Chuxing GAIA Initiative (https://sts.didiglobal.com).

# Package:
Tensorly: http://tensorly.org/stable/index.html [Note the effect of different unfolding or folding methods on tensor construction.]
