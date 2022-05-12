/*
Defines a base class for reward calculators for use in Iron Lab's Predictive Velocity Controller.
Subclasses from this base class should implement specific types of reward calculations.

Author: Connor Brooks
*/

#ifndef PREDIKCS_REWARD_CALCULATOR_H
#define PREDIKCS_REWARD_CALCULATOR_H

//includes
#include <vector>
#include <kdl/frames.hpp>
#include <Eigen/LU>
#include <boost/shared_ptr.hpp>

namespace predikcs
{

// forward declares
class RobotModel;
class MotionState;

class RewardCalculator
{
public:
    RewardCalculator()
    {
        dist_weight = -1.0;
        jerk_weight = -1.0;
        manip_weight = 1.0;
        lim_weight = -1.0;
    }
    ~RewardCalculator()
    {}

    static double GetLinearDistance(KDL::Vector* position_1, KDL::Vector* position_2);

    static double GetAngularDistance(KDL::Rotation* rotation_1, KDL::Rotation* rotation_2);

    static double CalculateDistance(KDL::Frame* frame_1, KDL::Frame* frame_2, double linear_distance_weight, double angular_distance_weight);

    static double CalculateSmoothness(std::vector<double>* old_accels, std::vector<double>* new_accels, double timestep);

    static double CalculateManipulability(boost::shared_ptr<RobotModel> robot_model, boost::shared_ptr<MotionState> candidate_motion);

    static double CalculateLimitCloseness(boost::shared_ptr<RobotModel> robot_model, boost::shared_ptr<MotionState> candidate_motion);

    double EvaluateMotionCandidate(boost::shared_ptr<RobotModel> robot_model, boost::shared_ptr<MotionState> old_state, boost::shared_ptr<MotionState> candidate_motion, KDL::Frame* ideal_position, bool verbose);

    double GetDistance(KDL::Frame* frame_1, KDL::Frame* frame_2);

    void SetParameters(double distance_weight, double jerk_weighting, double manipulability_weight, double limits_weight);

    double GetDistWeight(){ return dist_weight; }
    double GetJerkWeight(){ return jerk_weight; }
    double GetManipWeight(){ return manip_weight; }
    double GetLimWeight(){ return lim_weight; }

private:
    double dist_weight;
    double jerk_weight;
    double manip_weight;
    double lim_weight;
};

}

#endif  // PREDIKCS_REWARD_CALCULATOR_H