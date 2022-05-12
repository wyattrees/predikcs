/*
Defines a model class that handles a URDF of a robot used in planning.
Robot joint positions can be passed in for forward and inverse kinematics.
Built on KDL library.

Author: Connor Brooks
*/
#include "predikcs/robot_model.h"
#include <rclcpp/rclcpp.hpp>
#include <string>
#include <kdl/tree.hpp>
#include <kdl_parser/kdl_parser.hpp>


static const rclcpp::Logger LOGGER = rclcpp::get_logger("predikcs");

namespace predikcs
{

RobotModel::RobotModel(rclcpp::Node::SharedPtr node)
    : node_(node)
{
    busy_ = true;
    initialized_ = false;
    Init();
}

void RobotModel::Init()
{
    //Get root and tip joints for planning
    root_link_ = node_->get_parameter("/KCS_Controller/planning_root_link").as_string();
    tip_link_ = node_->get_parameter("/KCS_Controller/planning_tip_link").as_string();

    //Load URDF from parameter server
    std::string urdf_string = node_->get_parameter("/planning_robot_urdf").as_string();
    model_.initString(urdf_string);

    //Load KDL tree
    KDL::Tree kdl_tree;
    kdl_parser::treeFromUrdfModel(model_, kdl_tree);
    //Populate the chain
    kdl_tree.getChain(root_link_, tip_link_, kdl_chain_);

    jac_solver_ = std::make_shared<KDL::ChainJntToJacSolver>(kdl_chain_);
    jac_dot_solver_ = std::make_shared<KDL::ChainJntToJacDotSolver>(kdl_chain_);
    jnt_to_pos_solver_ = std::make_shared<KDL::ChainFkSolverPos_recursive>(kdl_chain_);
    jnt_pos_.resize(kdl_chain_.getNrOfJoints());
    jacobian_.resize(kdl_chain_.getNrOfJoints());
    jacobian_dot_.resize(kdl_chain_.getNrOfJoints());


    double new_joint_limit;
    for(int i = 0; i < kdl_chain_.getNrOfJoints(); ++i)
    {
        new_joint_limit = node_->get_parameter("/KCS_Controller/joint" + std::to_string(i) + "_pos_up_limit").as_double();
        jnt_pos_up_limits_.push_back(new_joint_limit);

        new_joint_limit = node_->get_parameter("/KCS_Controller/joint" + std::to_string(i) + "_pos_down_limit").as_double();
        jnt_pos_down_limits_.push_back(new_joint_limit);

        new_joint_limit = node_->get_parameter("/KCS_Controller/joint" + std::to_string(i) + "_vel_limit").as_double();
        jnt_vel_limits_.push_back(new_joint_limit);
        RCLCPP_INFO(LOGGER, "joint %d velocity limit: %.2f", i, jnt_vel_limits_[i]);
    }

    initialized_ = true;
    busy_ = false;
}

int RobotModel::GetNumberOfJoints()
{
    if(!initialized_){
        return -1;
    }

    return kdl_chain_.getNrOfJoints();
}

void RobotModel::GetJacobian(std::vector<double> joint_positions, Eigen::Matrix<double,6,Eigen::Dynamic>* jac)
{
    SetRobotPosition(joint_positions);
    jac->resize(jacobian_.rows(), jacobian_.columns());
    for(int i = 0; i < jacobian_.rows(); ++i)
    {
        for(int j = 0; j < jacobian_.columns(); ++j)
        {
            (*jac)(i, j) = jacobian_(i, j);
        }
    }
}

void RobotModel::GetJacobianDot(std::vector<double> joint_positions, int joint_index, Eigen::Matrix<double,6,Eigen::Dynamic>* jac_dot_)
{
    KDL::JntArray joint_pos(joint_positions.size());
    KDL::JntArray joint_vel(joint_positions.size());
    for(int i = 0; i < joint_positions.size(); ++i) {
        joint_pos(i) = joint_positions[i];
        if(i == joint_index) {
            joint_vel(i) = 1.0;
        } else {
            joint_vel(i) = 0.0;
        }
    }
    KDL::JntArrayVel system_state(joint_pos, joint_vel);
    jac_dot_solver_->JntToJacDot(system_state, jacobian_dot_);
    jac_dot_->resize(jacobian_dot_.rows(), jacobian_dot_.columns());
    for(int i = 0; i < jacobian_dot_.rows(); ++i)
    {
        for(int j = 0; j < jacobian_dot_.columns(); ++j)
        {
            (*jac_dot_)(i, j) = jacobian_dot_(i, j);
        }
    }
}

void RobotModel::GetPosition(std::vector<double> joint_positions, KDL::Frame* position)
{
    KDL::JntArray jnt_pos;
    jnt_pos.resize(joint_positions.size());
    //Update joint positions
    for(size_t i = 0; i < joint_positions.size(); i++)
    {
        jnt_pos(i) = joint_positions[i];
    }
    jnt_to_pos_solver_->JntToCart(jnt_pos, *position);

}

double RobotModel::GetJointPosUpLimit(int joint_index)
{
    return jnt_pos_up_limits_[joint_index];
}

double RobotModel::GetJointPosDownLimit(int joint_index)
{
    return jnt_pos_down_limits_[joint_index];
}

double RobotModel::GetJointVelLimit(int joint_index)
{
    return jnt_vel_limits_[joint_index];
}

//--------------------------------------------------------------------------------------------------
// Private functions

bool RobotModel::SetRobotPosition(std::vector<double> joint_positions)
{
    //Check that size of vector matches number of joints
    if(joint_positions.size() != GetNumberOfJoints())
    {
        return false;
    }

    //Update joint positions
    for(size_t i = 0; i < joint_positions.size(); i++)
    {
        jnt_pos_(i) = joint_positions[i];
    }

    //Update Jacobian
    jac_solver_->JntToJac(jnt_pos_, jacobian_);
    return true;
}

}