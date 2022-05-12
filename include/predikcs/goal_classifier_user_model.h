/*
Defines a user model class for use in Iron Lab's Predictive Velocity Controller.

Author: Connor Brooks
*/

#ifndef PREDIKCS_GOAL_CLASSIFIER_USER_MODEL_H
#define PREDIKCS_GOAL_CLASSIFIER_USER_MODEL_H

//includes
#include <vector>
#include <random>
#include <boost/shared_ptr.hpp>
#include <kdl/frames.hpp>
#include "user_model.h"
#include "reward_calculator.h"

namespace predikcs
{

// forward declares
class RobotModel;
class MotionState;

class GoalClassifierUserModel : public UserModel
{
public:

    GoalClassifierUserModel(int num_options, double action_timestep);

    void ClearSamplingState() override;

    void RandomSample(boost::shared_ptr<MotionState> state, std::vector<double>* sample) override;

    void SetRobotModel(boost::shared_ptr<RobotModel> robot_model) { robot_model_ = robot_model; }

    void SetGoals(const std::vector<std::vector<double>>& goal_points);

    void GetProbabilities(std::vector<double>& probabilities);

    void ResetProbabilities();

    void UpdateProbabilities(boost::shared_ptr<MotionState> state, const std::vector<double>& velocities);

private:
    boost::shared_ptr<RobotModel> robot_model_;
    std::vector<boost::shared_ptr<KDL::Frame>> goals;
    int sampling_goal;
    bool sampling_goal_set;
    std::vector<double> log_likelihoods;
};

}

#endif  // PREDIKCS_GOAL_CLASSIFIER_USER_MODEL_H