#include "FusionEKF.h"
#include <iostream>
#include "Eigen/Dense"
#include "tools.h"
#include "measurement_package.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;

/**
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;

  /**
   * -Finish initializing the FusionEKF.
   * -Set the process and measurement noises
   */
  /** The laser measurement matrix H - H is the matrix that projects your belief about the objects current state into the measurement space of the sensor.      * For Lidar, thisis a way of saying that we discard the velocity information from the state variable since Lidar only measures position.  The state vector    * 'x' contains information about [px,py,vx,vy] whereas the 'z' vector will only contain [px,py].  Multiplying 'H*x' allows us to compare 'x', our belief,    * with 'z' the sensor measurment. */
  
  H_laser_ << 1, 0, 0, 0, /* 2x4 matrix because x is a 4x1 matrix.  Mutiplying H*x should yield z a 1x2 matrix with only position information, not velocity*/
              0, 1, 0, 0;

  Hj_ << 1, 1, 0, 0, /* Jacobian Matrix is the derivative of h(x) with respect to x, it is a matrix containing all the partial derivatives */
         1, 1, 0, 0,
         1, 1, 1, 1; 

  // F_ is the initial state transition matrix, which when multiplied by x, predicts where the object will be after time delta_T
  ekf_.F_ = MatrixXd(4, 4);
  ekf_.F_ << 1, 0, 1, 0,
             0, 1, 0, 1,
             0, 0, 1, 0,
             0, 0, 0, 1;

  // P_ is the state covariance matrix, which contains information about the uncertainty of the objects position and velocity.
  ekf_.P_ = MatrixXd(4, 4);
  ekf_.P_ << 1, 0, 0, 0,
             0, 1, 0, 0,
             0, 0, 1000, 0,
             0, 0, 0, 1000;
}

/**
 * Destructor.
 */
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  /**********************************************************************************
   **							   Initialization								   **
   **********************************************************************************
   */
  if (!is_initialized_) {
    /**
     * -Initialize the state ekf_.x_ with the first measurement.
     * -Create the covariance matrix.
     * You'll need to convert radar from polar to cartesian coordinates.
     */

    // first measurement
    cout << "EKF: " << endl;
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      // TODO: Convert radar from polar to cartesian coordinates 
      // and initialize state.
      float rho     = measurement_pack.raw_measurements_(0);
      float phi     = measurement_pack.raw_measurements_(1);
      float rho_dot = measurement_pack.raw_measurements_(2);
      ekf_.x_(0) = rho     * cos(phi); /* px */
      ekf_.x_(1) = rho     * sin(phi); /* py */   
      ekf_.x_(2) = rho_dot * cos(phi); /* vx */
      ekf_.x_(3) = rho_dot * sin(phi); /* vy */
      }
    
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      // Initialize state for LIDAR.
	
      ekf_.x_(0) = measurement_pack.raw_measurements_(0); /* px */
      ekf_.x_(1) = measurement_pack.raw_measurements_(1); /* py */
      ekf_.x_(2) = 0; /* vx */
      ekf_.x_(3) = 0; /* vy */
      }
    
    previous_timestamp_ = measurement_pack.timestamp_;
    

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /**********************************************************************************
   **							   Prediction									   **
   **********************************************************************************
   */

  /**
   * Update the state transition matrix F according to the new elapsed time.
   * Time is measured in seconds.
   * TODO: Update the process noise covariance matrix.
   */
  
  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
  previous_timestamp_ = measurement_pack.timestamp_;
  
  // Update the state transition matrix F (so that time elapsed 1 second dt is included)
  ekf_.F_(0, 2) = dt;
  ekf_.F_(1, 3) = dt;
  
  // Update the process noise covariance matrix Q (so that time elapsed dt is included)
  float dt_2 = dt * dt;
  float dt_3 = dt_2 * dt;
  float dt_4 = dt_3 * dt;

  // Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
  float noise_ax = 9;
  float noise_ay = 9;

 // process covariance Q matrix
  ekf_.Q_ = MatrixXd(4, 4);
  ekf_.Q_ << dt_4/4*noise_ax, 0, dt_3/2*noise_ax, 0,
             0, dt_4/4*noise_ay, 0, dt_3/2*noise_ay,
             dt_3/2*noise_ax, 0, dt_2*noise_ax, 0,
             0, dt_3/2*noise_ay, 0, dt_2*noise_ay;
  
  // Call the EKF predict() function
  ekf_.Predict();
 

  /**********************************************************************************
   **							   Update										   **
   **********************************************************************************
   */

  /**
   * TODO:
   * - Use the sensor type to perform the update step.
   * - Update the state and covariance matrices.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // TODO: If Sensor Data is Radar Measurment updates
    
    // Jacobian
    Tools tools;
    Hj_ = MatrixXd(3, 4);
    Hj_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.Init(ekf_.x_, ekf_.P_, ekf_.F_, Hj_, R_radar_, ekf_.Q_);
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  }

  else {
    // If Sensor Data is LiDAR Sensor Measurement Updates
    ekf_.Init(ekf_.x_, ekf_.P_, ekf_.F_, H_laser_, R_laser_, ekf_.Q_);
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
