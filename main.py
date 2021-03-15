import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from math import *
from matplotlib.font_manager import FontProperties

font_family = 'DejaVu Sans'
title_font = FontProperties(family=font_family, style='normal', size=32, weight='normal', stretch='normal')
class KalmanData:


    def __init__(self, filePath, odom_cov=0.001):
        self.filePath = filePath
        time, values = self.get_data(filePath)

        #Time-series data
        self.time = time

        #Odometry/Encoder Data
        self.odom_x = values[:,0]
        self.odom_y = values[:,1]
        self.odom_theta = values[:,2]
        self.odom_cov = odom_cov

        #IMU data
        self.imu_heading = values[:,3]
        self.imu_heading_cov = values[:,4]

        #GPS data
        self.gps_x = values[:, 5]
        self.gps_y = values[:,6]
        self.gps_x_cov = values[:, 7]
        self.gps_y_cov = values[:, 8]

        #Misc Robot info
        self.velocity = 0.14 #Just some constant for now
        self.wheel_length = 1
        self.ang_velocity = self.velocity * tan(self.odom_theta[0])/self.wheel_length

    def get_data(self, file):
        dataFrame = pd.read_csv(file, skiprows=0)
        time = dataFrame.values[:, 0]
        values = dataFrame.values[:, 1:]

        return (time, values)

    def get_measurement(self, timeStep):
        return [self.odom_x[timeStep], self.odom_y[timeStep], self.velocity, self.odom_theta[timeStep], self.ang_velocity]

    def get_cov(self, timeStep):
        return [self.odom_cov, self.gps_x_cov, self.gps_y_cov]

    def plot_data(self):
        fig = plt.figure(figsize=[24,16])
        plt.title(f"T-series data from {self.filePath}", fontproperties=title_font )

        plt.plot(self.time, self.odom_x, label="odom_x")
        plt.plot(self.time, self.odom_y, label="odom_y")
        plt.plot(self.time, self.odom_theta, label="odom_theta")
        plt.plot(self.time, self.imu_heading, label="imu_heading")
        plt.plot(self.time, self.gps_x, label="gps_x")
        plt.plot(self.time, self.gps_y, label="gps_y")

        plt.legend()
        plt.show()
        plt.close(fig)

def apply_kalman_filter(kd):

    timeStep = kd.time[1] - kd.time[0]

    #Initialization
    x0 = np.array([kd.odom_x[0], kd.odom_y[0], kd.velocity, kd.odom_theta[0], kd.ang_velocity],dtype="float64")

    #State transition model
    '''Defined in the for loop'''

    #Observation model
    H = np.array( [[1, 0,0,0,0],
                   [0,1,0,0,0],
                   [0,0,1,0,0],
                   [0,0,0,1,0],
                   [0,0,0,0,1]
                   ]
            ,dtype="float64")
    #Covariance of process noise (just a guess, can tweak to heart's desire)
    Q = np.array( [ [0.00004, 0, 0, 0, 0],
                    [0, 0.00004, 0, 0, 0],
                    [0, 0, 0.0001, 0, 0],
                    [0, 0, 0, 0.0001, 0],
                    [0, 0, 0, 0, 0.0001]
                   ]
                  ,dtype="float64")
    #Covariance of observation Noise (just a guess, can tweak to heart's desire)
    R = np.array( [ [0.04, 0, 0, 0, 0],
                    [0, 0.04, 0, 0, 0],
                    [0, 0, 0.01, 0, 0],
                    [0, 0, 0, 0.01, 0],
                    [0, 0, 0, 0, 0.01]

                    ]
                  ,dtype="float64")
    #Control-input model
    B = np.array( [ [1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1]

                    ]
                  ,dtype="float64")
    #Control vector for B
    u = np.array([0,0,0,0,0], dtype="float64")

    #A posteriori estimate covariance matrix (estimated accuracy of state estimate xk)
    P = np.array( [ [0.01, 0, 0, 0, 0],
                    [0, 0.01, 0, 0, 0],
                    [0, 0, 0.01, 0, 0],
                    [0, 0, 0, 0.01, 0],
                    [0, 0, 0, 0, 0.01]
                   ]
                  ,dtype="float64")

    state = np.array([x0])
    uncertainty = np.array(P)

    for i in range(1, len(kd.time)):
        obs = np.array(kd.get_measurement(i),dtype="float64")
        F = np.array(
            [[1, 0, timeStep * cos(kd.odom_theta[i]), 0, 0],  # In essence, how does entry x change wrt other entries?
             [0, 1, timeStep * sin(kd.odom_theta[i]), 0, 0],
             [0, 0, 1, 0, 0],
             [0, 0, 0, 1, timeStep],
             [0, 0, 0, 0, 1]
             ]
            ,dtype="float64")

        #Prediction step
        xk = np.matmul(F, state[i-1]) + np.matmul(B, u)
        pk = np.matmul(F, uncertainty[i-1], F) + Q

        #Caluclate k-gain
        pm = np.matmul(H, pk, H.transpose()) + R
        K = np.matmul( pk, H.transpose(), (np.linalg.inv(pm)))

        #Refine predictions
        xk_f = xk + np.matmul(K, ( obs - np.matmul(H,xk)))
        pk_f = pk - np.matmul(K, H, pk)

        #Update step
        state = np.row_stack((state, xk_f))
        uncertainty = np.row_stack((uncertainty, pk_f))
    return state

EKF_DATA_circle = r"p1_files/EKF_DATA_circle.txt"
EKF_DATA_Rutgers = r"p1_files/EKF_DATA_Rutgers_ParkingLot.txt"

circleData = KalmanData(EKF_DATA_circle)

state = apply_kalman_filter(circleData)

fig = plt.figure(figsize=[24,16])

plt.title("Kalman filter",fontproperties=title_font)
plt.plot(circleData.time, state[:,0], label="Odom X")
plt.plot(circleData.time, state[:,1], label="Odom Y")
plt.plot(circleData.time, state[:,2], label="Velocity")
plt.plot(circleData.time, state[:,3], label="Odom Theta")
plt.plot(circleData.time, state[:,4], label="Angular Velocity")

plt.legend()
plt.show()
pass

