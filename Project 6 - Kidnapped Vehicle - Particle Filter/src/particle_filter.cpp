/**
 * particle_filter.cpp
 *
 * Author: Ian Whittal
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /** 
  *init function takes as input GPS coordinates double x and double y, initial heading estimate theta 
  *and an array of uncertainties for these measurements std[].
   * NOTE: Consult particle_filter.h for more information about this method (and others in this file).
   */
  num_particles = 100;  // Set the number of particles
  
  /**Set the number of particles. Initialize all particles to 
   * first position (based on estimates of x, y, theta and their uncertainties from GPS) and set all weights to 1. 
   */
  
  // Add random Gaussian noise to each particle using 
  std::default_random_engine gen;
  
  normal_distribution<double> init_x(0, std[0]);
  normal_distribution<double> init_y(0, std[1]);
  normal_distribution<double> init_theta(0, std[2]);
  
  // Generate particles
  for (int i=0; i<num_particles; i++){
    Particle particle;
    particle.id = i;
    particle.x = x; 
    particle.y = y;
    particle.theta = theta;
    particle.weight = 1.0;
   
    // Add random Gaussian noise to each particle using random engine "gen"
    std::default_random_engine gen;
    
    particle.x += init_x(gen); 
    particle.y += init_y(gen);
    particle.theta += init_theta(gen);
 
/** push_back() function is used to push elements into a vector from the back. The new value is inserted into the vector at the end, after the current 
*last element and the container size is increased by 1.
*/
    particles.push_back(particle);
    weights.push_back(particle.weight);
  }
  is_initialized = true;
}
  
// PREDICTION STEPS
void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
  * prediction takes as input the amount of time between timesteps delta_t the velocity and yaw rate measurment uncertainties std_pos[] and the current    
  * velocity and yaw_rate timestep measurments. Using these measurements the function will update the particles position estimates and account for sensor 
  * noise by adding Gaussian noise. Gaussian noise can be added by sampling from a Gaussian distribution with mean equal to the updated particle position 
  * and standard deviation equal to the standard deviation of the measurements.
  */
  // Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/

  // Add random Gaussian noise to each particle using randome engine gen
  std::default_random_engine gen;
  
  for (int i = 0; i < num_particles; i++) {

    // calculate new state from velocity, yaw rate and delta_t measurements using bicycle model to determine 
    // x and y components of final position for each particle
    // if yaw rate is zero, vehicle is straight
    if (fabs(yaw_rate) < 0.00001) {  
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    } 
    else {
      // vehicle is turning
      particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
      particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
      particles[i].theta += yaw_rate * delta_t;
    }
	// define normal distributions for sensor noise using std_pos[] velocity and yaw rate measurement noise
  	normal_distribution<double> norm_x(0, std_pos[0]);
  	normal_distribution<double> norm_y(0, std_pos[1]);
  	normal_distribution<double> norm_theta(0, std_pos[2]);
    
    // add random Gaussian noise to each particle using random engine gen
    particles[i].x += norm_x(gen);
    particles[i].y += norm_y(gen);
    particles[i].theta += norm_theta(gen);
  }
}

// UPDATE STEPS

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, vector<LandmarkObs>& observations)
{
  /**
  * dataAssociation takes as input two vectors of LandmarkObs objects; refer to helpers_fuctions .h for definition of this struct.
  * vector<LandmarkObs> predicted is the first vector which is prediction measurements between one particular particle and all of the map landmarks within 
  * sensor range vector<LandmarkObs>& observations is the actual landmark measurments gathered from the LIDAR Sensor. This function will perform nearest
  * neighbour data association and assign each sensor observation the map landmark ID associated with it.
  * Find the predicted measurement that is closest to each 
   * observed measurement and assign the observed measurement to this 
   * particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  unsigned int O = observations.size();
  unsigned int P = predicted.size();
  
  for (unsigned int i = 0; i< O; i++){
    
    // init distance to maximum possible
    double minDistance = std::numeric_limits<double>::max();
    int index_map = -1;
    for (unsigned int j=0; j<P; j++){
      double x_distance = observations[i].x - predicted[j].x;
      double y_distance = observations[i].y - predicted[j].y;
      double distance = x_distance * x_distance + y_distance * y_distance;
      // nearest neighbour
      if (distance < minDistance){
        minDistance = distance;
        index_map =  predicted[j].id;
      }
    }
    observations[i].id = index_map;
  }
}

// Update the weights of each particle using a multi-variate Gaussian distribution
double multiv_prob_gaussian(double sig_x, double sig_y, double x_obs, double y_obs,
                   double mu_x, double mu_y) {
  
  // calculate normalization term
  double gaussian_norm;
  gaussian_norm = 1 / (2 * M_PI * sig_x * sig_y);

  // calculate exponent
  double exponent;
  exponent = (pow(x_obs - mu_x, 2) / (2 * pow(sig_x, 2)))
               + (pow(y_obs - mu_y, 2) / (2 * pow(sig_y, 2)));

  // calculate weight using normalization terms and exponent
  double weight;
  weight = gaussian_norm * exp(-exponent);
    
  return weight;
}
  
void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
  * takes as input the range of the sensor range the landmark measurement uncertainties std_landmark[] a vector of landmark measurements  	  
  * vector<LandmarkObs> observations and map_landmarks as input.
   * Update the weights of each particle using a multi-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
for (unsigned int i = 0; i < particles.size(); ++i)
    {
    Particle particle = particles[i];
  	// probability set to 1.0
    double prob = 1.0;

        for (unsigned int j = 0; j < observations.size(); j++)
        {
            // Homogenous Transformation (transforms observations from vehicle's coordinate system to map's coordinate system)
            double x_map = particle.x + (cos(particle.theta) * observations[j].x) - (sin(particle.theta) * observations[j].y); // x_map >> transformed observation (TOBS) in x >> this means it is map coordinates
            double y_map = particle.y + (sin(particle.theta) * observations[j].x) + (cos(particle.theta) * observations[j].y); // y_map >> transformed observation (TOBS) in y >> this means it is map coordinates

          // Read map data from helper_functions.h use same nomenclature
            std::vector<Map::single_landmark_s> landmark_list = map_landmarks.landmark_list;
            double land_x; // x location of landmark
            double land_y; // y location of landmark
          	// maximum value will be 2x our sensor range
            double max_val = 2 * sensor_range;
            for (unsigned int k = 0; k < landmark_list.size(); k++)
            {
                // Calculate distance between particle and landmarks
                double local_land_x = landmark_list[k].x_f;
                double local_land_y = landmark_list[k].y_f;
              	// dist function
                double distance = dist(x_map, y_map, local_land_x, local_land_y);
                if ((distance <= sensor_range) && (distance <= max_val))
                {
                  // Calculate multivariate Gaussian normal distribution
                  land_x = local_land_x;
                  land_y = local_land_y;
                  max_val = distance;
                  prob = multiv_prob_gaussian(std_landmark[0], std_landmark[1], x_map, y_map, land_x, land_y);
                  particles[i].weight = prob;
                  weights[i] = prob;
                }
            }
        }
    }
}

void ParticleFilter::resample() {
  /**
   *  Resample particles with replacement with probability proportional 
   *  to their weight. Particles should regenerate close to the actual vehicle
   * function, use the weights of the particles in the particle filter and C++ standard libraries discrete_distribution function to update the 
   * particles to a Bayesian posterior distribution.
   *
   * NOTE: You may find std::discrete_distribution helpful here.
   * http://en.cppreference.com/w/cpp/random/discrete_distribution
   */
    std::default_random_engine gen;
    std::discrete_distribution<> d(weights.begin(), weights.end());
    std::vector<Particle> resampled_particles;

    for (int n = 0; n < num_particles; ++n) 
    {
        Particle particle = particles[d(gen)];
        resampled_particles.push_back(particle);
    }
    particles = resampled_particles;

}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}