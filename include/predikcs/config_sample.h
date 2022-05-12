/*
Defines a particular target joint configuration for use by a VOO Bandit object.
Author: Connor Brooks
*/

#ifndef PREDIKCS_CONFIG_SAMPLE_H
#define PREDIKCS_CONFIG_SAMPLE_H

// includes

#include <vector>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <boost/shared_ptr.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <rclcpp/rclcpp.hpp>
#include <random>
#include <kdl/frames.hpp>

namespace predikcs
{

// forward declares
class MotionState;
class RewardCalculator;
class RobotModel;
class UserModel;

class ConfigSample : public boost::enable_shared_from_this<ConfigSample>
{
public:
    ConfigSample(boost::shared_ptr<RobotModel> robot_model);
    ConfigSample(std::vector<double> target_joint_positions, boost::shared_ptr<RobotModel> robot_model, boost::shared_ptr<RewardCalculator> reward_calculator);
    ~ConfigSample()
    {}

    void ResetSample();

    double GetExpectedReward(int start_timestep, int num_timesteps, double discount_factor, std::vector<double>* joint_pos);

    void GenerateRollouts(boost::shared_ptr<MotionState> starting_state, int num_rollouts, int num_timesteps, double timestep_size, boost::shared_ptr<UserModel> user_model);

    double GetDistToSample(Eigen::VectorXd* sample);

    void GetJointVelocities(boost::shared_ptr<MotionState> joint_start_state, std::vector<double>* ee_command, std::vector<double>* joint_vels);

    std::vector<double> target_joint_pos;
private:
    bool samples_generated;
    bool null_sample;
    std::vector<std::vector<std::vector<double>>> predicted_joint_states_per_roll;
    std::vector<std::vector<double>> avg_timestep_scores_per_roll;
    boost::shared_ptr<RobotModel> robot_model_;
    boost::shared_ptr<RewardCalculator> reward_calculator_;
};

}

#endif  // PREDIKCS_CONFIG_SAMPLE_H