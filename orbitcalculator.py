import numpy as np


class orcal:

    def __init__(self):
        pass

    def orbit_normal_vector(self, inclination, raan):
        """
        Calculates the normal vector of the orbit plane.
        :param inclination: [rad] Inclination of the orbit.
        :param raan:        [rad] Right Ascension of Ascending Node (RAAN).
        :return:            Normal vector of the orbit plane.
        """
        # Convert angles from degrees to radians
        Om = raan
        i = inclination
        
        # Calculate the normal vector components
        n_vector = [
            np.sin(Om) * np.sin(i),
            -np.cos(Om) * np.sin(i),
            np.cos(i)
        ]
        return n_vector

    def anomaly_at_node(self, inclination, raan, arg_perigee, node_vector):
        """
        Calculates the true anomaly at the node.
        :param inclination: [rad] Inclination of the orbit.
        :param raan:        [rad] Right Ascension of Ascending Node (RAAN).
        :param arg_perigee: [rad] Argument of perigee.
        :param node_vector: Node vector in ECI coordinates.
        :return: [rad] True anomaly at the node.
        """
        R = self.rotation_matrix_eci2pqw(inclination, raan, arg_perigee) # Rotation matrix from ECI to PQW coordinates
        node_pqw = np.dot(R, node_vector)           # Transform node vector to PQW coordinates
        nu = np.arctan2(node_pqw[1], node_pqw[0])   # True anomaly in PQW coordinates
        return nu % (2*np.pi) # radian
        
    
    def calculate_orbit(self):
        # Placeholder for orbit calculation logic
        return f"Calculating orbit for {self.name} with radius {self.radius} km and period {self.period} days."
    

    def rotation_matrix_eci2pqw(self, inclination, raan, arg_perigee):
        """
        Constructs the rotation matrix from ECI to PQW coordinates.
        :param inclination: [rad] Inclination of the orbit in degrees.
        :param raan:        [rad] Right Ascension of Ascending Node (RAAN) in degrees.
        :param arg_perigee: [rad] Argument of perigee in degrees.
        :return: Rotation matrix from ECI to PQW coordinates.
        """
        # Convert angles from degrees to radians
        i = inclination
        Om = raan
        w = arg_perigee

        # Construct the rotation matrix
        R = np.array([
            [np.cos(Om) * np.cos(w) - np.sin(Om) * np.sin(w) * np.cos(i),
             np.cos(w) * np.sin(Om) + np.cos(Om) * np.sin(w) * np.cos(i),
             np.sin(w) * np.sin(i)],
            [-np.sin(Om) * np.cos(w) - np.cos(Om) * np.sin(w) * np.cos(i),
             -np.sin(Om) * np.sin(w) + np.cos(Om) * np.cos(w) * np.cos(i),
             np.cos(w) * np.sin(i)],
            [np.sin(Om) * np.sin(i),
             -np.cos(Om) * np.sin(i),
             np.cos(i)]
        ])
        return R
        
    def time_to_anomaly(self, eccentricity, mean_motion, current_ma, target_nu):
        """
        Calculates the time to reach a specific true anomaly.   
        :param eccentricity: Eccentricity of the orbit.
        :param mean_motion: [rev/days] Mean motion of the orbit.
        :param current_ma: [rad] Current mean anomaly.       
        :param target_nu: [rad] Target true anomaly in degrees.
        :return: Time in seconds to reach the target true anomaly.
        """
        e = eccentricity

        # find the mean anomaly corresponding to the target true anomaly
        nu = target_nu
        E = 2 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(nu / 2))  # Eccentric anomaly
        M = E - e * np.sin(E)  # Mean anomaly of the target true anomaly

        current_ma_rad = current_ma  # Convert current mean anomaly to radians
        delta_ma = M - current_ma_rad  # Change in mean anomaly
        if delta_ma < 0:        
            delta_ma += 2 * np.pi

        # Calculate the time to reach the target true anomaly
        time_to_target = delta_ma / (mean_motion * 2 * np.pi / 86400)  # Convert mean motion to rad/s and calculate time in seconds
        return time_to_target
        
        