#include <math.h>
#include <uWS/uWS.h>
#include <iostream>
#include <string>
#include "json.hpp"
#include "PID.h"

// for convenience
using nlohmann::json;
using std::string;

// Pi and functions for converting between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned, else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.find_last_of("]");
  if (found_null != string::npos) {
    return "";
  }
  else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 1);
  }
  return "";
}

int main() {
  uWS::Hub h;

  PID pid;
  /**
   * Initialize the pid variable.
   */
  
  // Tuning parameters (PID coefficients)
  // 1, 0, 0 - Result is vehicle osscilation increases with each period until running off the road
  // 1, 1, 0 - Vehicle begins hard left turn in a circle immediately after start
  // 1, 0, 1 - Vehicle begins hard right turn into curb immediately after start
  // 0.2, 0.004, 3.0 - Coefficients from Lecture -  Vehicle is able to drive forward but with hard/sharp steering corrections
  // the derivative component countersteers the vehicle and helps not to overshoot the trajectoy and not to oscillate, let's increase the "d" component
  // 0.2, 0.004, 3.0 - Good performance, vehicle is able to navigate test track, lots of oscillation in turns, need to limit CTE error in turns, increase "d" component
  // 0.2, 0.004, 4.5 - Better performance, not sure if increasing helped in the turns though, response strength in turns still high, lower "p" component
  // 0.15, 0.004, 3.0 - Better performance, better performance in turns, need to retest effect of "d" component, increase "d" component
  // 0.14, 0.004, 3.0 - Best performance, still slight issues in turns, tune "I" parameter
  // 0.14, 0.003, 3.0 - Wheels chalked right at start.
  // 0.14, 0.003, 4.0 - Wheels touched red of Apex on final turn
  // 0.14, 0.004, 4.0 - Appears to drive successfully, reduce P and D by factor of half.
  // 0.07, 0.004, 2.0 - Wheels chalked right at start. 
  // 0.07, 0.004, 1.5 - Best performance - Slight oscillation most likely due to Proportional controller - lower "P"
 

  pid.Init(0.06, 0.004, 1.5);
  
  h.onMessage([&pid](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length, 
                     uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    
    if (length && length > 2 && data[0] == '4' && data[1] == '2') {
      auto s = hasData(string(data).substr(0, length));

      if (s != "") {
        auto j = json::parse(s);

        string event = j[0].get<string>();

        if (event == "telemetry") {
          // j[1] is the data JSON object
          double cte = std::stod(j[1]["cte"].get<string>());
          // double speed = std::stod(j[1]["speed"].get<string>());
          // double angle = std::stod(j[1]["steering_angle"].get<string>());
          double steer_value;
          
          /**
           * Calculate steering value here. Steering value is between [-1, 1].
           */
          
          // Update error and calculate steer_value at each step
          pid.UpdateErrors(cte);
          steer_value = pid.TotalError(); // remember: steer value ( = steer angle α ) equals (-p_error * Kp -i_error * Ki - d_error * Kd) which is the sum of the three errors weighted by their corresponding coefficient

          std::cout << "CTE: " << cte << " Steering Value: " << steer_value << std::endl;
          
          // ensuring steer value is between [-1, 1]
          if (steer_value > 1) steer_value = 1;
          else if (steer_value < -1) steer_value = -1;
        
          json msgJson;
          msgJson["steering_angle"] = steer_value;
          msgJson["throttle"] = 0.3;
          auto msg = "42[\"steer\"," + msgJson.dump() + "]";
          std::cout << msg << std::endl;
          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }  // end "telemetry" if
      } else {
        // Manual driving
        string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }  // end websocket message if
  }); // end h.onMessage

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code, 
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  
  h.run();
}