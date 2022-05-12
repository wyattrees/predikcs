/*
Defines a base class for user models for use in Iron Lab's Predictive Velocity Controller.
Subclasses from this base class should implement specific types of probabilistic user models.

Author: Connor Brooks
*/

#ifndef PREDIKCS_USER_MODEL_H
#define PREDIKCS_USER_MODEL_H

//includes
#include <vector>
#include <random>
#include <boost/shared_ptr.hpp>

namespace predikcs
{

// forward declares
class RobotModel;
class MotionState;

class UserModel
{
public:
    UserModel(int num_options, double action_timestep);
    
    virtual ~UserModel()
    {}

    virtual void ClearSamplingState(){ SetLastSampledCommand(last_user_velocity_command_); }

    virtual void RandomSample(boost::shared_ptr<MotionState> state, std::vector<double>* sample);

    void SetNumOptions(int num_options){ num_options_ = num_options; }

    void SetActionTimestep(double action_timestep){ action_timestep_ = action_timestep; }

    void SetLastVelocityCommand(std::vector<double> last_velocity_command) { 
        last_user_velocity_command_ = last_velocity_command;
        SetLastSampledCommand(last_user_velocity_command_);
    }

protected:
    void SetLastSampledCommand(std::vector<double> last_sampled_command) {
        last_velocity_command_ = last_sampled_command;
        double vel_norm = 0.0;
        for(int i = 0; i < last_velocity_command_.size(); ++i)
        {
            vel_norm += pow(last_velocity_command_[i], 2);
        }
        last_velocity_command_norm_ = sqrt(vel_norm);
    }

    int num_options_;
    double action_timestep_;
    std::vector<double> last_velocity_command_;
    std::vector<double> last_user_velocity_command_;
    double last_velocity_command_norm_;
    std::random_device rd;
    std::default_random_engine random_generator;
    std::uniform_real_distribution<double> random_distribution;
};

}

#endif  // PREDIKCS_USER_MODEL_H