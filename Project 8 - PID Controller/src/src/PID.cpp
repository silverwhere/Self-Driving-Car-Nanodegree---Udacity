#include "PID.h"

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp_, double Ki_, double Kd_) {
  /**
   * Initialize PID coefficients and errors
   */
  
  // coefficients
  Kp = Kp_; // proportional coefficient
  Ki = Ki_; // integral coefficient
  Kd = Kd_; // differential coefficient
  
  // errors
  p_error = 0; // error in the proportional term equals the CTE (cross-track-error)
  i_error = 0; // error in the integral term equals the sum of the CTE over time
  d_error = 0; // error in the differential term equals the difference between actual CTE and previous CTE
}

void PID::UpdateErrors(double cte) {
  /**
   * Update PID errors based on cte.
   */
  
  d_error = cte - p_error; // difference between actual and previous error. Denominator (delta t) equals 1 --> not needed
  p_error = cte;
  i_error += cte;
}

double PID::TotalError() {
  /**
   * Calculate and return the total error which is the steer value ( = steer angle α )
   */
  return -p_error * Kp -i_error * Ki - d_error * Kd;
}