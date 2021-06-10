from lowpass import LowPassFilter
from yaw_controller import YawController
from pid import PID
import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704

'''
Low pass filter, filters out high frequency noise in the velocity messages

'''

class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband, decel_limit,
                accel_limit, wheel_radius, wheel_base, steer_ratio, max_lat_accel, max_steer_angle):
        
        
        self.yaw_controller = YawController(wheel_base, steer_ratio, 0.1, max_lat_accel, max_steer_angle)

        # Determining parameters experimentally
        kp = 0.3
        ki = 0.1
        kd = 0.0
        mn = 0. # Minimum throttle value
        mx = 0.2 # maximum throttle value
        # PID is called from PID.py
        self.throttle_controller = PID(kp, ki, kd, mn, mx)

        tau = 0.5 # 1/(2pi*tau) = cutoff frequency
        ts = 0.02
        self.vel_lpf = LowPassFilter(tau,ts)

        #Based on Udacity Vehicle Carla
        self.vehicle_mass = vehicle_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit # comfort parameter to ensure gentle deceleration
        self.accel_limit = accel_limit # comfort parameter to ensure gentle acceleration
        self.wheel_radius = wheel_radius

        self.last_time = self.print_time = rospy.get_time()

        # Control gets called in dbw.py
        # dbw_enabled is to prevent the integral term in a PID controller from accumulating error should the car be stopped at a traffic light
    def control(self, current_vel, dbw_enabled, linear_vel, angular_vel):
        if not dbw_enabled:
            # Reset the controller
            self.throttle_controller.reset()
            return 0., 0., 0.
        
        # Filter the high frequency velocities off
        current_vel = self.vel_lpf.filt(current_vel)

        steering = self.yaw_controller.get_steering(linear_vel,angular_vel,current_vel)

        # Calculate the error for the PID controller
        vel_err = linear_vel - current_vel
        self.last_vel = current_vel

        #ros get_time() used to get the sample time for each step of the PID controller 
        current_time = rospy.get_time()
        sample_time = current_time - self.last_time
        self.last_time = current_time

        throttle = self.throttle_controller.step(vel_err, sample_time)
        brake = 0

        # Velocity error, where do we want to be vs. where are we actually
        if linear_vel == 0. and current_vel < 0.1:
            throttle = 0
            brake = 700 #N*m to hold the car in place if we are stopped at a light. Acceleration ~ 1m/s^2


        elif throttle < 0.1 and vel_err < 0:
            throttle = 0
            decel = max(vel_err, self.decel_limit)
            brake = abs(decel) * self.vehicle_mass * self.wheel_radius # Torque N*m

        return throttle, brake, steering