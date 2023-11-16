#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "eigen3/Eigen/Eigen"
// #include "eigen3/Eigen/LU"
// #include "eigen3/Eigen/SVD"
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include "pinocchio/algorithm/jacobian.hxx"
#include "pinocchio/multibody/fwd.hpp"
#include "my_controller_interface/srv/my_controller.hpp"

#include "control_msgs/action/follow_joint_trajectory.hpp"
#include "control_msgs/msg/joint_trajectory_controller_state.hpp"
#include "controller_interface/controller_interface.hpp"
#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "rclcpp/duration.hpp"
#include "rclcpp/subscription.hpp"
#include "rclcpp/time.hpp"
#include "rclcpp/timer.hpp"
#include "rclcpp_lifecycle/lifecycle_publisher.hpp"
#include "rclcpp_lifecycle/node_interfaces/lifecycle_node_interface.hpp"
#include "realtime_tools/realtime_buffer.h"
#include "trajectory_msgs/msg/joint_trajectory.hpp"
#include "trajectory_msgs/msg/joint_trajectory_point.hpp"

namespace my_controller
{
    class MyController : public controller_interface::ControllerInterface
    {
    public:
        CONTROLLER_INTERFACE_PUBLIC
        MyController();
        ~MyController() = default;

        void initDesiredPose(const Eigen::Vector3d &position_d_target,
                             const Eigen::Quaterniond &orientation_d_target);

        /*! \brief Sets the nullspace configuration without using filtering
         *
         * \param[in] q_d_nullspace_target Nullspace joint positions
         */
        void initNullspaceConfig(const Eigen::VectorXd &q_d_nullspace_target);

        /*! \brief Sets the number of joints
         *
         * \param[in] n_joints Number of joints
         */
        void setNumberOfJoints(size_t n_joints);

        /*! \brief Set the desired diagonal stiffnessess + nullspace stiffness
         *
         * \param[in] stiffness Stiffnesses: position, orientation, nullspace
         * \param[in] auto_damping Apply automatic damping
         */
        void setStiffness(const Eigen::Matrix<double, 7, 1> &stiffness, bool auto_damping = true);

        /*! \brief Sets the Cartesian and nullspace stiffnesses
         *
         * \param[in] t_x Translational stiffness x
         * \param[in] t_y Translational stiffness y
         * \param[in] t_z Translational stiffness z
         * \param[in] r_x Rotational stiffness x
         * \param[in] r_y Rotational stiffness y
         * \param[in] r_z Rotational stiffness z
         * \param[in] n   Nullspace stiffness
         * \param[in] auto_damping Apply automatic damping
         */
        void setStiffness(double t_x, double t_y, double t_z, double r_x, double r_y, double r_z, double n, bool auto_damping = true);

        /*! \brief Sets the Cartesian and nullspace stiffnesses
         *
         * \param[in] t_x Translational stiffness x
         * \param[in] t_y Translational stiffness y
         * \param[in] t_z Translational stiffness z
         * \param[in] r_x Rotational stiffness x
         * \param[in] r_y Rotational stiffness y
         * \param[in] r_z Rotational stiffness z
         * \param[in] auto_damping Apply automatic damping
         */
        void setStiffness(double t_x, double t_y, double t_z, double r_x, double r_y, double r_z, bool auto_damping = true);

        /*! \brief Set the desired damping factors
         *
         * \param[in] t_x Translational damping x
         * \param[in] t_y Translational damping y
         * \param[in] t_z Translational damping z
         * \param[in] r_x Rotational damping x
         * \param[in] r_y Rotational damping y
         * \param[in] r_z Rotational damping z
         * \param[in] n   Nullspace damping
         */
        void setDampingFactors(double d_x, double d_y, double d_z, double d_a, double d_b, double d_c, double d_n);

        /*! \brief Sets the desired end-effector pose
         *
         * Sets them as a new target, so filtering can be applied on them.
         * \param[in] position_d New reference position
         * \param[in] orientation_d New reference orientation
         */
        void setReferencePose(const Eigen::Vector3d &position_d, const Eigen::Quaterniond &orientation_d);

        /*! \brief Sets a new nullspace joint configuration
         *
         * Sets them as a new target, so filtering can be applied on them.
         * \param[in] q_d_nullspace_target New joint configuration
         */
        void setNullspaceConfig(const Eigen::VectorXd &q_d_nullspace_target);

        /*! \brief Sets filtering on stiffness + end-effector pose.
         *
         * Default inactive && depends on update_frequency
         * \param[in] update_frequency The expected controller update frequency
         * \param[in] filter_params_nullspace_config Filter setting for nullspace config
         * \param[in] filter_params_nullspace_config Filter setting for the stiffness
         * \param[in] filter_params_nullspace_config Filter setting for the pose
         * \param[in] filter_params_nullspace_config Filter setting for the commanded wrenc
         */
        void setFiltering(double update_frequency, double filter_params_nullspace_config, double filter_params_stiffness, double filter_params_pose,
                          double filter_params_wrench);

        /*! \brief Maximum commanded torque change per time step
         *
         * Prevents too large changes in the commanded torques by using saturation.
         * \param[in] d Torque change per timestep
         */
        void setMaxTorqueDelta(double d);

        /*! \brief Sets maximum commanded torque change per time step and the update frequency
         *
         * Prevents too large changes in the commanded torques by using saturation.
         * \param[in] d Torque change per timestep
         * \param[in] update_frequency Update frequency
         */
        void setMaxTorqueDelta(double d, double update_frequency);

        /*! \brief Apply a virtual Cartesian wrench in the root frame (often "world")
         *
         * Prevents too large changes in the commanded torques by using saturation.
         * \param[in] cartesian_wrench Wrench to apply
         */
        void applyWrench(const Eigen::Matrix<double, 6, 1> &cartesian_wrench);

        /*! \brief Returns the commanded torques. Performs a filtering step.
         *
         * This function assumes that the internal states have already been updates. The it utilizes the control rules to calculate commands.
         * \return Eigen Vector of the commanded torques
         */
        Eigen::VectorXd calculateCommandedTorques();

        /*! \brief Returns the commanded torques. Performs a filtering step and updates internal state.
         *
         * This function utilizes the control rules.
         * \param[in] q Joint positions
         * \param[in] dq Joint velocities
         * \param[in] position End-effector position
         * \param[in] orientation End-effector orientation
         * \param[in] jacobian Jacobian
         */
        Eigen::VectorXd calculateCommandedTorques(const Eigen::VectorXd &q, const Eigen::VectorXd &dq,
                                                  const Eigen::Vector3d &position, Eigen::Quaterniond orientation,
                                                  const Eigen::MatrixXd &jacobian);

        /*! \brief Get the state of the controller. Updates when "calculateCommandedTorques" is called
         *
         * \param[out] q Joint positions
         * \param[out] dq Joint velocities
         * \param[out] position End-effector position
         * \param[out] orientation End-effector orientation
         * \param[out] position_d End-effector reference position
         * \param[out] orientation_d End-effector reference orientation
         * \param[out] cartesian_stiffness Cartesian stiffness
         * \param[out] nullspace_stiffness Nullspace stiffness
         * \param[out] q_d_nullspace Nullspace reference position
         * \param[out] cartesian_damping Cartesian damping
         */
        void getState(Eigen::VectorXd *q, Eigen::VectorXd *dq, Eigen::Vector3d *position, Eigen::Quaterniond *orientation,
                      Eigen::Vector3d *position_d, Eigen::Quaterniond *orientation_d,
                      Eigen::Matrix<double, 6, 6> *cartesian_stiffness, double *nullspace_stiffness,
                      Eigen::VectorXd *q_d_nullspace, Eigen::Matrix<double, 6, 6> *cartesian_damping) const;

        /*! \brief Get the state of the controller. Updates when "calculateCommandedTorques" is called
         *
         * \param[out] position_d End-effector reference position
         * \param[out] orientation_d End-effector reference orientation
         * \param[out] cartesian_stiffness Cartesian stiffness
         * \param[out] nullspace_stiffness Nullspace stiffness
         * \param[out] q_d_nullspace Nullspace reference position
         * \param[out] cartesian_damping Cartesian damping
         */
        void getState(Eigen::Vector3d *position_d, Eigen::Quaterniond *orientation_d,
                      Eigen::Matrix<double, 6, 6> *cartesian_stiffness, double *nullspace_stiffness,
                      Eigen::VectorXd *q_d_nullspace, Eigen::Matrix<double, 6, 6> *cartesian_damping) const;

        /*! \brief Get the currently applied commands
         *
         * \return Eigen Vector with commands
         */
        Eigen::VectorXd getLastCommands() const;

        /*! \brief Get the currently applied Cartesian wrench
         *
         * \return Eigen Vector with the applied wrench
         */
        Eigen::Matrix<double, 6, 1> getAppliedWrench() const;

        /*! \brief Get the current pose error
         *
         * \return Eigen Vector with the pose error for translation and rotation
         */
        Eigen::Matrix<double, 6, 1> getPoseError() const;

        /*! \brief Updates the trajectory.
         *
         * Called periodically from the update function if a trajectory is running.
         * A trajectory is run by going through it point by point, calculating forward kinematics and applying
         * the joint configuration to the nullspace control.
         */
        void trajUpdate();

        /*! \brief Updates the state based on the joint handles.
         *
         * Gets latest joint positions, velocities and efforts and updates the forward kinematics as well as the Jacobian.
         */
        void updateState();

        /*! \brief Initializes trajectory handling
         *
         * Subscribes to joint trajectory topic and starts the trajectory action server.
         * \param[in] nh Nodehandle
         * \return Always true.
         */
        void initTrajectory(const std::shared_ptr<my_controller_interface::srv::MyController::Request> request,
                            std::shared_ptr<my_controller_interface::srv::MyController::Response> response);

        CONTROLLER_INTERFACE_PUBLIC
        controller_interface::InterfaceConfiguration command_interface_configuration() const override;

        CONTROLLER_INTERFACE_PUBLIC
        controller_interface::InterfaceConfiguration state_interface_configuration() const override;

        CONTROLLER_INTERFACE_PUBLIC
        controller_interface::return_type update(
            const rclcpp::Time &time, const rclcpp::Duration &period) override;

        CONTROLLER_INTERFACE_PUBLIC
        controller_interface::CallbackReturn on_init() override;

        CONTROLLER_INTERFACE_PUBLIC
        controller_interface::CallbackReturn on_configure(
            const rclcpp_lifecycle::State &previous_state) override;

        CONTROLLER_INTERFACE_PUBLIC
        controller_interface::CallbackReturn on_activate(
            const rclcpp_lifecycle::State &previous_state) override;

        CONTROLLER_INTERFACE_PUBLIC
        controller_interface::CallbackReturn on_deactivate(
            const rclcpp_lifecycle::State &previous_state) override;

        CONTROLLER_INTERFACE_PUBLIC
        controller_interface::CallbackReturn on_cleanup(
            const rclcpp_lifecycle::State &previous_state) override;

        CONTROLLER_INTERFACE_PUBLIC
        controller_interface::CallbackReturn on_error(
            const rclcpp_lifecycle::State &previous_state) override;

        CONTROLLER_INTERFACE_PUBLIC
        controller_interface::CallbackReturn on_shutdown(
            const rclcpp_lifecycle::State &previous_state) override;

    protected:
        size_t n_joints_{7}; //!< Number of joints to control

        Eigen::Matrix<double, 6, 6> cartesian_stiffness_{Eigen::Matrix<double, 6, 6>::Identity()}; //!< Cartesian stiffness matrix
        Eigen::Matrix<double, 6, 6> cartesian_damping_{Eigen::Matrix<double, 6, 6>::Identity()};   //!< Cartesian damping matrix

        Eigen::VectorXd q_d_nullspace_;          //!< Current nullspace reference pose
        Eigen::VectorXd q_d_nullspace_target_;   //!< Nullspace reference target pose
        double nullspace_stiffness_{0.0};        //!< Current nullspace stiffness
        double nullspace_stiffness_target_{0.0}; //!< Nullspace stiffness target
        double nullspace_damping_{0.0};          //!< Current nullspace damping
        double nullspace_damping_target_{0.0};   //!< Nullspace damping target

        Eigen::Matrix<double, 6, 6> cartesian_stiffness_target_{Eigen::Matrix<double, 6, 6>::Identity()}; //!< Cartesian stiffness target
        Eigen::Matrix<double, 6, 6> cartesian_damping_target_{Eigen::Matrix<double, 6, 6>::Identity()};   //!< Cartesian damping target
        Eigen::Matrix<double, 7, 1> damping_factors_{Eigen::Matrix<double, 7, 1>::Ones()};                //!< Damping factors

        Eigen::VectorXd q_;     //!< Joint positions
        Eigen::VectorXd dq_;    //!< Joint velocities
        Eigen::VectorXd tau_m_; //!< Measured joint torques

        Eigen::MatrixXd jacobian_; //!< Jacobian. Row format: 3 translations, 3 rotation

        // End Effector
        Eigen::Matrix<double, 6, 1> error_;                          //!< Calculate pose error
        Eigen::Vector3d position_{Eigen::Vector3d::Zero()};          //!< Current end-effector position
        Eigen::Vector3d position_d_{Eigen::Vector3d::Zero()};        //!< Current end-effector reference position
        Eigen::Vector3d position_d_target_{Eigen::Vector3d::Zero()}; //!< End-effector target position

        Eigen::Quaterniond orientation_{Eigen::Quaterniond::Identity()};          //!< Current end-effector orientation
        Eigen::Quaterniond orientation_d_{Eigen::Quaterniond::Identity()};        //!< Current end-effector target orientation
        Eigen::Quaterniond orientation_d_target_{Eigen::Quaterniond::Identity()}; //!< End-effector orientation target

        //  External applied forces
        Eigen::Matrix<double, 6, 1> cartesian_wrench_{Eigen::Matrix<double, 6, 1>::Zero()};        //!< Current Cartesian wrench
        Eigen::Matrix<double, 6, 1> cartesian_wrench_target_{Eigen::Matrix<double, 6, 1>::Zero()}; //!< Cartesian wrench target

        Eigen::VectorXd tau_c_; //!< Last commanded torques

        double update_frequency_{1000};              //!< Update frequency in Hz
        double filter_params_nullspace_config_{1.0}; //!< Nullspace filtering
        double filter_params_stiffness_{1.0};        //!< Cartesian stiffness filtering
        double filter_params_pose_{1.0};             //!< Reference pose filtering
        double filter_params_wrench_{1.0};           //!< Commanded wrench filtering

        double delta_tau_max_{1.0}; //!< Maximum allowed torque change per time step

        // Trajectory

        pinocchio::Model model_; // model of the robot

        bool traj_running_{false};                         //!< True when running a trajectory
        trajectory_msgs::msg::JointTrajectory trajectory_; //!< Currently played trajectory
        unsigned int traj_index_{0};                       //!< Index of the current trajectory point

        std::vector<std::string> joint_names_;
        std::vector<std::string> command_interface_types_;
        std::vector<std::string> state_interface_types_;

        rclcpp::Service<my_controller_interface::srv::MyController>::SharedPtr trajectory_service_;
        realtime_tools::RealtimeBuffer<std::shared_ptr<trajectory_msgs::msg::JointTrajectory>>
            traj_msg_external_point_ptr_;
        bool new_msg_ = false;
        rclcpp::Time start_time_;
        std::shared_ptr<trajectory_msgs::msg::JointTrajectory> trajectory_msg_;
        trajectory_msgs::msg::JointTrajectoryPoint point_interp_;

        std::vector<std::reference_wrapper<hardware_interface::LoanedCommandInterface>>
            joint_effort_command_interface_;
        std::vector<std::reference_wrapper<hardware_interface::LoanedStateInterface>>
            joint_position_state_interface_;
        std::vector<std::reference_wrapper<hardware_interface::LoanedStateInterface>>
            joint_velocity_state_interface_;
        std::vector<std::reference_wrapper<hardware_interface::LoanedStateInterface>>
            joint_effort_state_interface_;

        std::unordered_map<
            std::string, std::vector<std::reference_wrapper<hardware_interface::LoanedCommandInterface>> *>
            command_interface_map_ = {
                {"effort", &joint_effort_command_interface_}};

        std::unordered_map<
            std::string, std::vector<std::reference_wrapper<hardware_interface::LoanedStateInterface>> *>
            state_interface_map_ = {
                {"position", &joint_position_state_interface_},
                {"velocity", &joint_velocity_state_interface_},
                {"effort", &joint_effort_state_interface_}};

    private:
        /*! \brief Implements the damping based on a stiffness
         *
         * Damping rule is 2*sqrt(stiffness)
         * \param[in] stiffness Stiffness value
         * \return Damping value
         */
        double dampingRule(double stiffness) const;

        /*! \brief Applies the damping rule with all stiffness values
         */
        void applyDamping();

        /*! Sets the update frequency
         *
         * \param[in] freq Update frequency
         */
        void setUpdateFrequency(double freq);

        /*! \brief Sets the filter value and asserts bounds
         *
         * \param[in] val New value
         * \param[out] saved_val Pointer to the value to be set
         */
        void setFilterValue(double val, double *saved_val);

        /*! \brief Adds a percental filtering effect to the nullspace configuration
         *
         * Gradually moves the nullspace configuration to the target configuration.
         */
        void updateFilteredNullspaceConfig();

        /*! \brief Adds a percental filtering effect to stiffness
         */
        void updateFilteredStiffness();

        /*! \brief Adds a percental filtering effect to the end-effector pose
         */
        void updateFilteredPose();

        /*! \brief Adds a percental filtering effect to the applied Cartesian wrench
         */
        void updateFilteredWrench();

        /*! \brief Get forward kinematics solution.
         *
         * Calls RBDyn to get the forward kinematics solution.
         * \param[in]  q            Joint position vector
         * \param[out] position     End-effector position
         * \param[out] orientation  End-effector orientation
         * \return Always true.
         */
        bool getFk(const Eigen::VectorXd &q, Eigen::Vector3d *position, Eigen::Quaterniond *rotation) const;

        /*! \brief Get Jacobian from RBDyn
         *
         * Gets the Jacobian for given joint positions and joint velocities.
         * \param[in]  q         Joint position vector
         * \param[in]  dq        Joint velocity vector
         * \param[out] jacobian  Calculated Jacobian
         * \return True on success, false on failure.
         */
        bool getJacobian(const Eigen::VectorXd &q, Eigen::MatrixXd *jacobian);
    };
} // namespace my_controller