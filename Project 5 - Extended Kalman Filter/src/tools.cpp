#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
   * Calculate the Root Mean Squared Error (RMSE) to track performance - Estimated Values and Ground Truth values.  The difference gives us a resididual.
   * The residual is then squared, then averaged and then the square root gives the error metric.  Lower error = higher estimation accuracy.  
   */

VectorXd rmse(4);
  rmse << 0, 0, 0, 0;

  // check the validity of the following inputs:
  //  * the estimation vector size should not be zero
  //  * the estimation vector size should equal ground truth vector size
  if (estimations.size() != ground_truth.size()
      || estimations.size() == 0){
      std::cout << "Invalid estimation or ground_truth data" << std::endl;
      return rmse;
  }

  //accumulate squared residuals
  for (unsigned int i = 0; i < estimations.size(); ++i){

      VectorXd residual = estimations[i] - ground_truth[i];

      //coefficient-wise multiplication
      residual = residual.array()*residual.array();
      rmse += residual;
  }

  //calculate the mean
  rmse = rmse / estimations.size();

  //calculate the squared root
  rmse = rmse.array().sqrt();

  //return the result
  return rmse;
}


/**
* Calculate the Jacobian Matrix, this tool will be used to derive a linear approximation for the 'h' function, from the taylor series we will only keep the  * expansion up to the Jacobian Matrix Df(a).  We will ignore the Hessan matrix D^2f(a) and other higher order terms. Assuming (x-a) is small, (x-a)^2 or the * multi-dimensional equivalent (x-a)^T*(x-a) will be even smaller; the EKF we'll be using assumes that the higher order terms beyond the Jacobian are        * negligible.*/

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {

  MatrixXd Hj(3,4);
  // recover state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  // pre-compute a set of terms to avoid repeated calculation
  float c1 = px*px+py*py;
  float c2 = sqrt(c1);
  float c3 = (c1*c2);

  //check division by zero
  if (fabs(c1) < 0.0001){
      std::cout << "CalculateJacobian () - Error - Division by Zero" << std::endl;
      return Hj;
  }

  // compute the Jacobian matrix
  Hj << (px/c2), (py/c2), 0, 0,
      -(py/c1), (px/c1), 0, 0,
      py*(vx*py - vy*px)/c3, px*(px*vy - py*vx)/c3, px/c2, py/c2;

  return Hj;
}

