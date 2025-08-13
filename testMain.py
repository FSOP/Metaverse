from time import timezone
import numpy as np
from sgp4.api import Satrec, jday
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import propTLE
from scipy.interpolate import interp1d
from scipy.signal import argrelextrema

# Example TLE
sat1 = ["YAOGAN-36 02C",
"1 54045U 22133D   25223.80796962  .00013396  00000+0  44861-3 0  9997",
"2 54045  34.9939 265.4105 0014387 237.9985 121.9334 15.30012790157550"]

sat2 = ["STARLINK-32299",
        "1 61702U 24195R   25223.83091972 -.00003649  00000+0 -11111-3 0  9991",
        "2 61702  53.1613  49.9539 0002069 148.0413 212.0713 15.30198280 45129"]

str_tca = "2025-08-13 15:54:44.515"
# str_start_time = "2025-08-13 15:50:00.000"
# str_end_time = "2025-08-13 16:00:00.000"

step_size = 1 # seconds

tca = datetime.strptime(str_tca, "%Y-%m-%d %H:%M:%S.%f")
start_time = (tca - timedelta(hours=1))
stop_time = (tca + timedelta(hours=1)) 


ep1 = propTLE.propTLE(sat1[1], sat1[2], start_time, stop_time, step_size)
ep2 = propTLE.propTLE(sat2[1], sat2[2], start_time, stop_time, step_size)


diff = ep1.ephemeris[:,1:4] - ep2.ephemeris[:,1:4]  # Example operation on ephemeris data
dist = np.linalg.norm(diff, axis=1)  # Calculate the distance between the two satellites at each time step

# plt.plot(ep1.ephemeris[:,0], dist)
# plt.xlabel("Unix Time")
# plt.ylabel("Distance (km)")
# plt.title("Distance Between Satellites Over Time")
# plt.savefig("distance_plot.png")  # Saves the plot as a PNG file

# Find indices of local minima
min_indices = argrelextrema(dist, np.less)[0]

# Prepare lists for refined minima
refined_times = []
refined_dists = []

for idx in min_indices:
    t_min = ep1.ephemeris[idx, 0]
    t_grid = np.arange(t_min - 60, t_min + 60, 0.001)  # ±1 min, 0.001 sec steps

    f_interp = interp1d(ep1.ephemeris[:,0], dist, kind='cubic')
    dist_fine = f_interp(t_grid)

    min_fine_idx = np.argmin(dist_fine)
    tca_unix = t_grid[min_fine_idx]
    tca_dist = dist_fine[min_fine_idx]

    refined_times.append(tca_unix)
    refined_dists.append(tca_dist)

# Plot the distance curve
plt.plot(ep1.ephemeris[:,0], dist, label="Distance")

# Mark local minima (original)
plt.scatter(ep1.ephemeris[min_indices,0], dist[min_indices], color='red', marker='o', label="Local Minima")

# Mark refined closest approach points
plt.scatter(refined_times, refined_dists, color='blue', marker='x', label="Refined Closest Approach")

plt.xlabel("Unix Time")
plt.ylabel("Distance (km)")
plt.title("Distance Between Satellites Over Time")
plt.legend()
plt.savefig("distance_plot.png")

##
idx_tca = min_indices[np.argmin(refined_dists)]
r1_vec = ep1.ephemeris[idx_tca, 1:4]  # position of sat1 at TCA
r2_vec = ep2.ephemeris[idx_tca, 1:4]  # position of sat2 at TCA
v1_vec = ep1.ephemeris[idx_tca, 4:7]  # velocity of sat1 at TCA
v2_vec = ep2.ephemeris[idx_tca, 4:7]  # velocity of sat2 at TCA

# Relative position and velocity
rel_pos = r2_vec - r1_vec
rel_vel = v2_vec - v1_vec

# Normalize velocity for encounter plane
intrack = rel_vel / np.linalg.norm(rel_vel)
radial = rel_pos / np.linalg.norm(rel_pos)
crosstrack = np.cross(intrack, radial)
crosstrack /= np.linalg.norm(crosstrack)

# Build encounter plane basis
R = np.vstack([radial, intrack, crosstrack]).T  # 3x3 matrix

# Project relative position into encounter plane
rel_pos_plane = R.T @ rel_pos

# Use assumed uncertainties (meters)
sigma_r = 0.1      # radial
sigma_i = 0.3      # in-track
sigma_c = 0.1      # cross-track

# Covariance matrix in encounter plane
cov_plane = np.diag([sigma_r**2, sigma_i**2, sigma_c**2])

# Combined hard-body radius (meters)
r1 = 0.005
r2 = 0.005
r_combined = r1 + r2

# Projected separation in encounter plane (usually use in-track and cross-track)
sep_intrack = rel_pos_plane[1]
sep_crosstrack = rel_pos_plane[2]

# 2D covariance ellipse axes
a = sigma_i
b = sigma_c

# 2D separation
d_2d = np.sqrt(sep_intrack**2 + sep_crosstrack**2)

# Maximum probability formula (Alfano, 2D, encounter plane)
if d_2d > r_combined:
    P_max = np.exp(-0.5 * (d_2d**2) / (a * b)) * (r_combined / (np.sqrt(2 * np.pi * a * b)))
else:
    P_max = 1.0

print(f"Maximum collision probability (encounter plane): {P_max:.2e}")
print("end")

