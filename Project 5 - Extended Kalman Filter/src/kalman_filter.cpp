#include "kalman_filter.h"
#include "tools.h"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;

/* 
 * Please note that the Eigen library does not initialize 
 *   VectorXd or MatrixXd objects with zeros upon creation.
 */

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in; /* x is the mean state vector. For an extended Kalman filter, the mean state vector contains information about the object’s position and velocity that you are tracking. It is called the “mean” state vector because position and velocity are represented by a gaussian distribution with mean x. */
  P_ = P_in; /* P is the state covariance matrix, which contains information about the uncertainty of the object’s position and velocity. You can think of it as containing standard deviations. */
  F_ = F_in; /* F is a matrix that when multiplied with x, predicts where the object after time delta-T */
  H_ = H_in; /* H is the matrix that projects your belief about the object’s current state into the measurement space of the sensor. For lidar, this is a fancy way of saying that we discard velocity information from the state variable since the lidar sensor only measures position: The state vector x contains information about [px,py,vx,vy] whereas the z vector will only contain [px,py]. Multiplying Hx allows us to compare x, our belief, with z, the sensor measurement.*/
  R_ = R_in; /* R is the covariance matrix that represents uncertainty in our sensor measurements.  The time dimensions of the R matrix is squared and each side of its matrix is the same length as the number of measurement parameters. */
  Q_ = Q_in; /* Q is the process covariance matrix to model the stochastic part of the state transition function. */
}

void KalmanFilter::Predict() {
  /**
   * Predict the state
   */
  
  x_ = F_ * x_; /* State Transition Function */
  MatrixXd Ft = F_.transpose(); /* Transpose to Multiply */
  P_ = F_ * P_ * Ft + Q_; /* State covariance Matrix */
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
   * Update the state by using Kalman Filter equations (LIDAR)
   */
   /**
     * KF Measurement update step
     */
    VectorXd z_pred = H_ * x_; /* measurement vector 2 parameters */
    VectorXd y = z - z_pred; /* Measurement Error */
    MatrixXd Ht = H_.transpose(); /* Measurement Function Transpose */
    MatrixXd S = H_ * P_ * Ht + R_; /* Measurement Prediction Covariance */
    MatrixXd Si = S.inverse(); /* Inverse to Multiply */
    MatrixXd PHt = P_ * Ht; /* Break out Multiplication */
    MatrixXd K = PHt * Si; /* Kalman Gain */

    // new state
    x_ = x_ + (K * y); /* Best Estimate */
  	long x_size = x_.size(); /* size modifier */
    MatrixXd I = MatrixXd::Identity(x_size, x_size); /* Identity matrix, used for matrix inversion */
    P_ = (I - K * H_) * P_; /* Uncertainty Covariance */
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
   * update the state by using Extended Kalman Filter equations (RADAR)
   * for radar there is no H matrix that will map the state vector x into Polar coordinates; instead you need to calculate the mapping
   * manually to convert from cartesian coordinates to polar coordinates.
   */
    float px = x_(0);
    float py = x_(1);
    float vx = x_(2);
    float vy = x_(3);

    //readings convert from cartesian coordinates to polar coordinates.
    float rho = sqrt((px * px) + (py * py)); /** The range (rho) is the distance to the pedestrian. The range is basically the magnitude of                                                                 * the position vector (P) which is defined as p = sqrt((px * px) + (py * py)) */
  	float phi = atan2(py, px); /* The bearing (phi) is the angle between P and X referenced CCW from the x-axis. */
  	float rho_dot; /* The radial velocity (rho_dot) is the project of the velocity v on the line */
  
  // check if rho is zero	
  if (fabs(rho) < 0.0001) 
  {
    	rho_dot = 0;
  } 
  else 
  {
    rho_dot = (px*vx + py*vy)/rho;
  }
  
  
  VectorXd z_pred(3); /* measurement vector 3 parameters */
  z_pred << rho, 
            phi, 
            rho_dot;
  
  VectorXd y = z - z_pred; /* Measurement Error */
  
  /* angle normalization Normalizing Angles (Description from Wolfgang-Stefani and Knowledge hub
  * In C++, atan2() returns values between -pi and pi.
  * When calculating phi in y = z - h(x) for radar measurements,
  * the resulting angle phi in the y vector should be adjusted so that it is between -pi and pi.
  * The Kalman filter is expecting small angle values between the range -pi and pi.
  * When working in radians, you can add 2π or subtract 2π until the angle is within the desired range
  */ 
  if (y(1) < -M_PI) // y(1) refers to phi
    {
      y(1) += 2 * M_PI;
    }
    
    else if (y(1) > M_PI)
    {
      y(1) -= 2 * M_PI;
    }
  
  MatrixXd Ht = H_.transpose(); /* Measurement Function */
  MatrixXd S = H_ * P_ * Ht + R_; /* Measurement Prediction Covariance */
  MatrixXd Si = S.inverse(); /* Inverse to Multiply */
  MatrixXd PHt = P_ * Ht; 
  MatrixXd K = PHt * Si; /* Kalman Gain */

  //new estimate
  x_ = x_ + (K * y); /* Best Estimate */
  long x_size = x_.size(); /* size modifier */
  MatrixXd I = MatrixXd::Identity(x_size, x_size); /* Identity matrix, used for matrix inversion */
  P_ = (I - K * H_) * P_; /* Uncertainty Covariance */
}