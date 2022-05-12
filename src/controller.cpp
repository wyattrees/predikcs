#include <rclcpp/rclcpp.hpp>
#include <vector>
#include <boost/format.hpp>
#include <boost/pointer_cast.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <control_msgs/msg/gripper_command.hpp>
#include <string>
#include <std_msgs/msg/string.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <yaml-cpp/yaml.h>
#include "predikcs/robot_model.h"
#include "predikcs/user_model.h"
#include "predikcs/goal_classifier_user_model.h"
#include "predikcs/reward_calculator.h"
#include "predikcs/motion_state.h"
#include "predikcs/voo_bandit.h"

using namespace std::chrono_literals;
using std::placeholders::_1;

static const rclcpp::Logger LOGGER = rclcpp::get_logger("predikcs");

class Controller
{
public:
    Controller(std::shared_ptr<rclcpp::Node> nh, double loop_rate) : nh_(nh), voo_spec(new predikcs::VooSpec), robot(new predikcs::RobotModel(nh_)),
    user(new predikcs::UserModel(0,0)), reward(new predikcs::RewardCalculator)
    {
        joint_sub = nh_->create_subscription<sensor_msgs::msg::JointState>("joint_States", 1, std::bind(&Controller::JointUpdateCallback, this, _1));
        vel_command_sub = nh_->create_subscription<geometry_msgs::msg::Twist>("teleop_commands", 1, std::bind(&Controller::VelocityCommandCallback, this, _1));
        gripper_command_sub = nh_->create_subscription<control_msgs::msg::GripperCommand>("/gripper_controller/gripper_action/goal", 1, std::bind(&Controller::GripperCommandCallback, this, _1));
        command_pub = nh_->create_publisher<sensor_msgs::msg::JointState>("joint_commands", 1);

        // Set search parameters. These can be defined as needed but should be tuned for your use case
        // Longer rollouts will typically provide smoother movement for a given time interval, provided there is enough computational resources to gather a large number of samples (100+ active samples at any time)
        // Reward parameters can be configured to emphasize different components (or you can define a reward model that is entirely custom!)
        baseline = false;
        update_goal_probabilities = false;
        ReadParams();
        voo_spec->sampling_loop_rate = loop_rate;
        voo_spec->sampling_time_limit = loop_rate - (loop_rate / 10.0);
        voo_spec->max_voronoi_samples = 10;
        voo_bandit = boost::shared_ptr<predikcs::VooBandit>(new predikcs::VooBandit(voo_spec, robot, reward));

        current_fetch_command_msg = new sensor_msgs::msg::JointState();

        last_velocity_command = std::vector<double>(6, 0);
        last_joint_positions = std::vector<double>(no_of_joints, 0);
        last_joint_velocities = std::vector<double>(no_of_joints, 0);
        last_joint_accelerations = std::vector<double>(no_of_joints, 0);
        current_fetch_command_msg->velocity = std::vector<double>(no_of_joints, 0);
        last_joint_msg_time = 0;
        max_accel_factor = 4.0;
        vel_command_waiting = false;
        PublishControllerSpec();
    }

    ~Controller()
    {}

    void PublishControllerSpec()
    {
        if(baseline)
        {
            nh_->set_parameter(rclcpp::Parameter("/teleop_type", "baseline"));
        }
        else
        {
            nh_->set_parameter(rclcpp::Parameter("/teleop_type", (boost::format("Sample %.1f, Tau %d, Rolls %d, Roll Steps %d, Delta T %.1f, Gamma %.1f, Reward Params %.1f, %.1f, %.1f, %.1f") % voo_spec->uniform_sample_prob % voo_spec->tau % voo_spec->sample_rollouts % voo_spec->rollout_steps % voo_spec->delta_t % voo_spec->gamma % reward->GetDistWeight() % reward->GetJerkWeight() % reward->GetManipWeight() % reward->GetLimWeight()).str()));
        }
    }

    void ReadParams()
    {
        no_of_joints = nh_->get_parameter("/KCS_Controller/number_of_joints").as_int();
        joint_names.clear();
        std::string new_joint_name;
        double new_joint_limit;
        for(int i = 0; i < no_of_joints; ++i)
        {
            new_joint_name = nh_->get_parameter("/KCS_Controller/joint" + std::to_string(i) + "_name").as_string();
            joint_names.push_back(new_joint_name);
        }

        voo_spec->uniform_sample_prob = nh_->get_parameter("/KCS_Controller/voo_spec/uniform_sample_prob").as_double();
        voo_spec->tau = nh_->get_parameter("/KCS_Controller/voo_spec/tau").as_int();
        voo_spec->sample_rollouts = nh_->get_parameter("/KCS_Controller/voo_spec/sample_rollouts").as_int();
        voo_spec->rollout_steps = nh_->get_parameter("/KCS_Controller/voo_spec/rollout_steps").as_int();
        voo_spec->delta_t = nh_->get_parameter("/KCS_Controller/voo_spec/delta_t").as_double();
        voo_spec->gamma = nh_->get_parameter("/KCS_Controller/voo_spec/temporal_discount").as_double();
        double dist_weight, jerk_weight, manip_weight, lim_weight;

        dist_weight = nh_->get_parameter("/KCS_Controller/reward_params/distance").as_double();
        jerk_weight = nh_->get_parameter("/KCS_Controller/reward_params/jerk").as_double();
        manip_weight = nh_->get_parameter("/KCS_Controller/reward_params/manipulability").as_double();
        lim_weight = nh_->get_parameter("/KCS_Controller/reward_params/limits").as_double();
        reward->SetParameters(dist_weight, jerk_weight, manip_weight, lim_weight);

        user_model_type = nh_->get_parameter("/KCS_Controller/user_model").as_int();
        if(user_model_type == 0)
        {
            user = boost::shared_ptr<predikcs::UserModel>(new predikcs::UserModel(1, voo_spec->delta_t));
        }
        else if (user_model_type == 1)
        {
            boost::shared_ptr<predikcs::GoalClassifierUserModel> goal_user(new predikcs::GoalClassifierUserModel(1, voo_spec->delta_t));
            goal_user->SetRobotModel(robot);
            int num_goals = nh_->get_parameter("KCS_Controller/num_goals").as_int();
            std::vector<std::vector<double>> goals;
            for(int i = 0; i < num_goals; ++i)
            {
                std::vector<double> goal_points;
                nh_->declare_parameter("/KCS_Controller/goal_" + std::to_string(i + 1), rclcpp::ParameterType::PARAMETER_DOUBLE);
                goal_points[i] = nh_->get_parameter("/KCS_Controller/goal_" + std::to_string(i + 1)).as_double();
                goals.push_back(goal_points);
            }
            goal_user->SetGoals(goals);
            user = goal_user;
            update_goal_probabilities = true;
        }
        else
        {
            baseline = true;
        }
    }

    void JointUpdateCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
    {
        float k_min_update_time = 0.025;
        std::vector<double> new_joint_positions;
        std::vector<double> new_joint_velocities;
        std::vector<double> new_joint_accelerations;
        double joint_msg_time = (double)msg->header.stamp.sec;
        if(joint_msg_time - last_joint_msg_time < k_min_update_time || msg->position.size() < joint_names.size())
        {
            return;
        }
        int j = 0;
        for(int i = 0; i < msg->position.size(); ++i)
        {
            if(j < last_joint_positions.size() && msg->name[i].compare(joint_names[j]) == 0)
            {
                new_joint_positions.push_back(msg->position[i]);
                double joint_dist = new_joint_positions[j] - last_joint_positions[j];
                if(abs(joint_dist) > M_PI && robot->GetJointPosUpLimit(j) == std::numeric_limits<double>::infinity())
                {
                    joint_dist = copysign(abs(joint_dist - (2 * M_PI)), -1 * joint_dist);
                }
                new_joint_velocities.push_back(joint_dist / (joint_msg_time - last_joint_msg_time));
                new_joint_accelerations.push_back((new_joint_velocities[j] - last_joint_velocities[j]) / (joint_msg_time - last_joint_msg_time));
                ++j;
            }
        }
        if(new_joint_positions.size() != joint_names.size())
        {
            // Bad message, discard
            RCLCPP_ERROR(LOGGER, "Improperly sized joint positions (missing joint names)");
            return;
        }
        last_joint_positions = new_joint_positions;
        last_joint_velocities = new_joint_velocities;
        last_joint_accelerations = new_joint_accelerations;
        last_joint_msg_time = joint_msg_time;
    }

    void VelocityCommandCallback(const geometry_msgs::msg::Twist::SharedPtr msg)
    {
        std::vector<double> new_velocity_command;
        new_velocity_command.push_back(msg->linear.x);
        new_velocity_command.push_back(msg->linear.y);
        new_velocity_command.push_back(msg->linear.z);
        new_velocity_command.push_back(msg->angular.x);
        new_velocity_command.push_back(msg->angular.y);
        new_velocity_command.push_back(msg->angular.z);
        last_velocity_command = new_velocity_command;
        vel_command_waiting = true;
    }

    void GripperCommandCallback(const control_msgs::msg::GripperCommand::SharedPtr msg)
    {
        if(update_goal_probabilities){
            static_cast<predikcs::GoalClassifierUserModel*>(user.get())->ResetProbabilities();
        }
    }

    void DecayCommand()
    {
        double k_min_vel_command = 0.01;
        double k_decay_factor = 1.25;
        // Exponentially decay velocity command
        for(int i = 0; i < current_fetch_command_msg->velocity.size(); ++i)
        {
            if(abs(current_fetch_command_msg->velocity[i]) < k_min_vel_command)
            {
                current_fetch_command_msg->velocity[i] = 0.0;
            }
            else
            {
                current_fetch_command_msg->velocity[i] = current_fetch_command_msg->velocity[i] / k_decay_factor;
            }
        }
    }

    void GetNewCommand()
    {
        bool all_zeros = true;
        for(int i = 0; i < last_velocity_command.size(); ++i)
        {
            if(last_velocity_command[i] != 0.0)
            {
                all_zeros = false;
                break;
            }
        }

        if(all_zeros || !vel_command_waiting)
        {
            /*
            boost::shared_ptr<predikcs::MotionState> current_state( new predikcs::MotionState(last_joint_positions, last_joint_velocities, last_joint_accelerations) );
            current_state->CalculatePosition(robot);
            double quat_x, quat_y, quat_z, quat_w;
            current_state->position.M.GetQuaternion(quat_x, quat_y, quat_z, quat_w);
            ROS_ERROR("Current state: %.2f, %.2f, %.2f | %.2f, %.2f, %.2f, %.2f", current_state->position.p.x(), current_state->position.p.y(), current_state->position.p.z(), quat_x, quat_y, quat_z, quat_w);
            */
            DecayCommand();
        }
        else if (baseline)
        {
            boost::shared_ptr<predikcs::MotionState> current_state( new predikcs::MotionState(last_joint_positions, last_joint_velocities, last_joint_accelerations) );
            current_state->CalculatePosition(robot);
            double quat_x, quat_y, quat_z, quat_w;
            current_state->position.M.GetQuaternion(quat_x, quat_y, quat_z, quat_w);
            RCLCPP_ERROR(LOGGER, "Current state: %.2f, %.2f, %.2f | %.2f, %.2f, %.2f, %.2f", current_state->position.p.x(), current_state->position.p.y(), current_state->position.p.z(), quat_x, quat_y, quat_z, quat_w);
            voo_bandit->GetBaselineJointVelocities(current_state, &last_velocity_command, &(current_fetch_command_msg->velocity));
        }
        else
        {
            user->SetLastVelocityCommand(last_velocity_command);

            // Find best current null movement
            boost::shared_ptr<predikcs::MotionState> current_state( new predikcs::MotionState(last_joint_positions, last_joint_velocities, last_joint_accelerations) );
            
            current_state->CalculatePosition(robot);
            double quat_x, quat_y, quat_z, quat_w;
            current_state->position.M.GetQuaternion(quat_x, quat_y, quat_z, quat_w);
            RCLCPP_ERROR(LOGGER, "Current state: %.2f, %.2f, %.2f | %.2f, %.2f, %.2f, %.2f", current_state->position.p.x(), current_state->position.p.y(), current_state->position.p.z(), quat_x, quat_y, quat_z, quat_w);
            
            voo_bandit->GetCurrentBestJointVelocities(current_state, &last_velocity_command, &(current_fetch_command_msg->velocity));
            if(update_goal_probabilities){
                static_cast<predikcs::GoalClassifierUserModel*>(user.get())->UpdateProbabilities(current_state, last_velocity_command);
                
                std::vector<double> current_probs;
                static_cast<predikcs::GoalClassifierUserModel*>(user.get())->GetProbabilities(current_probs);
                RCLCPP_ERROR(LOGGER, "Current probs: %.2f %.2f %.2f %.2f %.2f %.2f", current_probs[0], current_probs[1], current_probs[2], current_probs[3], current_probs[4], current_probs[5]);
                
            }
        }
        PublishCommand();
        vel_command_waiting = false;
    }

    void GenerateNewSamples()
    {
        // Add more samples while waiting for next command
        boost::shared_ptr<predikcs::MotionState> next_state( new predikcs::MotionState(&last_joint_positions, &last_joint_velocities, &(current_fetch_command_msg->velocity), voo_spec->sampling_loop_rate, robot, 0.0) );
        
        voo_bandit->GenerateSamples(next_state, user);
    }

    void TrimAccel()
    {
        double max_accel = max_accel_factor * voo_spec->sampling_loop_rate;
        for(int i = 0; i < current_fetch_command_msg->velocity.size(); ++i)
        {
            if(current_fetch_command_msg->velocity[i] == 0.0)
            {
                continue;
            }
            if(abs(current_fetch_command_msg->velocity[i] - last_joint_velocities[i]) > max_accel)
            {
                current_fetch_command_msg->velocity[i] = last_joint_velocities[i] + copysign(max_accel, current_fetch_command_msg->velocity[i] - last_joint_velocities[i]);
            }
        }        
    }

    void PublishCommand()
    {
        TrimAccel();
        command_pub->publish(*(current_fetch_command_msg));
    }

    boost::shared_ptr<predikcs::VooSpec> voo_spec;
    boost::shared_ptr<predikcs::VooBandit> voo_bandit;

private:
    std::shared_ptr<rclcpp::Node> nh_;
    int user_model_type;
    int no_of_joints;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr vel_command_sub;
    rclcpp::Subscription<control_msgs::msg::GripperCommand>::SharedPtr gripper_command_sub;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr command_pub;
    std::vector<double> last_joint_positions;
    std::vector<double> last_joint_velocities;
    std::vector<double> last_joint_accelerations;
    double last_joint_msg_time;
    double max_accel_factor;
    std::vector<double> last_velocity_command;
    std::vector<std::string> joint_names;
    std::vector<double> joint_vel_limits;
    boost::shared_ptr<predikcs::RobotModel> robot;
    boost::shared_ptr<predikcs::UserModel> user;
    boost::shared_ptr<predikcs::RewardCalculator> reward;
    bool vel_command_waiting;
    bool update_goal_probabilities;
    bool baseline;
    std::string controller_spec;
    sensor_msgs::msg::JointState* current_fetch_command_msg;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);

    auto node = rclcpp::Node::make_shared("KCS_Controller");
    std::map<std::string, std::string> str_params {
            {"/KCS_Controller/planning_root_link", ""},
            {"/KCS_Controller/planning_tip_link", ""},
            {"/KCS_Controller/joint0_name", ""},
            {"/KCS_Controller/joint1_name", ""},
            {"/KCS_Controller/joint2_name", ""},
            {"/KCS_Controller/joint3_name", ""},
            {"/KCS_Controller/joint4_name", ""},
            {"/KCS_Controller/joint5_name", ""},
            {"/KCS_Controller/joint6_name", ""}

        };
    node->declare_parameters<std::string>("", str_params);
    std::map<std::string, double> double_params {
        {"/KCS_Controller/joint0_pos_up_limit", 0.0},
        {"/KCS_Controller/joint0_pos_down_limit", 0.0},
        {"/KCS_Controller/joint0_vel_limit", 0.0},
        {"/KCS_Controller/joint1_pos_up_limit", 0.0},
        {"/KCS_Controller/joint1_pos_down_limit", 0.0},
        {"/KCS_Controller/joint1_vel_limit", 0.0},
        {"/KCS_Controller/joint2_pos_up_limit", 0.0},
        {"/KCS_Controller/joint2_pos_down_limit", 0.0},
        {"/KCS_Controller/joint2_vel_limit", 0.0},
        {"/KCS_Controller/joint3_pos_up_limit", 0.0},
        {"/KCS_Controller/joint3_pos_down_limit", 0.0},
        {"/KCS_Controller/joint3_vel_limit", 0.0},
        {"/KCS_Controller/joint4_pos_up_limit", 0.0},
        {"/KCS_Controller/joint4_pos_down_limit", 0.0},
        {"/KCS_Controller/joint4_vel_limit", 0.0},
        {"/KCS_Controller/joint5_pos_up_limit", 0.0},
        {"/KCS_Controller/joint5_pos_down_limit", 0.0},
        {"/KCS_Controller/joint5_vel_limit", 0.0},
        {"/KCS_Controller/joint6_pos_up_limit", 0.0},
        {"/KCS_Controller/joint6_pos_down_limit", 0.0},
        {"/KCS_Controller/joint6_vel_limit", 0.0},
        {"/KCS_Controller/voo_spec/uniform_sample_prob", 0.0},
        {"/KCS_Controller/voo_spec/delta_t", 0.0},
        {"/KCS_Controller/voo_spec/temporal_discount", 0.0},
        {"/KCS_Controller/reward_params/distance", 0.0},
        {"/KCS_Controller/reward_params/jerk", 0.0},
        {"/KCS_Controller/reward_params/manipulability", 0.0},
        {"/KCS_Controller/reward_params/limits", 0.0}
        };
    node->declare_parameters<double>("", double_params);

    std::map<std::string, int> int_params {
        {"/KCS_Controller/number_of_joints", 0},
        {"/KCS_Controller/voo_spec/tau", 0},
        {"/KCS_Controller/voo_spec/sample_rollouts", 0},
        {"/KCS_Controller/voo_spec/rollout_steps", 0},
        {"/KCS_Controller/user_model", 0},
        {"/KCS_Controller/num_goals", 0},
        
    };
    node->declare_parameters<int>("", int_params);
    rclcpp::sleep_for(1000ms);

    Controller controller(node, 0.1);
    rclcpp::sleep_for(1000ms);


    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(node);
    

    // ros::AsyncSpinner spinner(1);
    rclcpp::Rate loop_rate(10);

    // spinner.start();
    while(rclcpp::ok())
    {
        executor.spin_once();
        controller.GetNewCommand();
        controller.GenerateNewSamples();
        executor.spin_once();
        loop_rate.sleep();
    }
}