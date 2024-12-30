import pandas as pd
import networkx as nx
import numpy as np
from sklearn.preprocessing import StandardScaler

# Step 1: Load PMU data and topology information
pmu_data = pd.read_csv('/path/to/pmu_data.csv')
topology = nx.read_gml('/path/to/topology.gml')

# Step 2: Perform anomaly detection (classification step - you can use LightGBM/XGBoost)
# For simplicity, assume 'anomaly_score' column represents detection result
anomalous_buses = pmu_data[pmu_data['anomaly_score'] > 0.5]  # threshold value

# Step 3: Triangulate event based on topology and anomaly scores
def calculate_distance(bus1, bus2):
    # Simplified example: Euclidean distance or any other metric
    return np.linalg.norm(bus1 - bus2)

# Find connected buses in the topology
event_location_scores = []

for bus in anomalous_buses['bus_id']:
    # Get connected buses (neighbors in the topology graph)
    neighbors = list(topology.neighbors(bus))
    
    for neighbor in neighbors:
        # Calculate anomaly score based on the data from both buses
        distance = calculate_distance(pmu_data[pmu_data['bus_id'] == bus]['anomaly_score'],
                                     pmu_data[pmu_data['bus_id'] == neighbor]['anomaly_score'])
        event_location_scores.append((bus, neighbor, distance))

# Step 4: Localize event - aggregate or select the bus with the lowest distance score
localized_event = min(event_location_scores, key=lambda x: x[2])  # bus with lowest distance
print("Localized Event Origin: Bus", localized_event[0])

# Step 5: Visualize event location on the grid
import matplotlib.pyplot as plt
nx.draw(topology, with_labels=True)
plt.scatter([localized_event[0]], [localized_event[1]], color='red')  # mark the localized event
plt.show()
