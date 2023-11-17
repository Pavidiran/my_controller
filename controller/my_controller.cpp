#include "my_controller/my_controller.hpp"
#include "my_controller/pseudo_inversion.hpp"

#include <stddef.h>
#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include <functional>

#include "rclcpp/qos.hpp"
#include "rclcpp/time.hpp"
#include "rclcpp_lifecycle/node_interfaces/lifecycle_node_interface.hpp"
#include "rclcpp_lifecycle/state.hpp"

using config_type = controller_interface::interface_configuration_type;

namespace my_controller
{
  MyController::MyController() : controller_interface::ControllerInterface()
  {

    this->setStiffness(200., 200., 200., 20., 20., 20., 0.);
    this->cartesian_stiffness_ = this->cartesian_stiffness_target_;
    this->cartesian_damping_ = this->cartesian_damping_target_;
  }

  Eigen::Vector3d calculateOrientationError(const Eigen::Quaterniond &orientation_d, Eigen::Quaterniond orientation)
  {
    // Orientation error
    if (orientation_d.coeffs().dot(orientation.coeffs()) < 0.0)
    {
      orientation.coeffs() << -orientation.coeffs();
    }
    // "difference" quaternion
    const Eigen::Quaterniond error_quaternion(orientation * orientation_d.inverse());
    // convert to axis angle
    Eigen::AngleAxisd error_quaternion_angle_axis(error_quaternion);
    return error_quaternion_angle_axis.axis() * error_quaternion_angle_axis.angle();
  }

  /*! \brief Calculates a filtered percental update
   *
   * \param[in] target Target value
   * \param[in] current Current value
   * \param[in] filter Percentage of the target value
   * \return Calculated value
   */
  template <typename T>
  inline T filteredUpdate(T target, T current, double filter)
  {
    return (1.0 - filter) * current + filter * target;
  }

  /*! \brief Calculates the filter step
   *
   * \param[in] update_frequency   Update frequency in Hz
   * \param[in] filter_percentage  Filter percentage
   * \return Filter step
   */
  inline double filterStep(const double &update_frequency, const double &filter_percentage)
  {
    const double kappa = -1 / (std::log(1 - std::min(filter_percentage, 0.999999)));
    return 1.0 / (kappa * update_frequency + 1.0);
  }

  /*! \brief Saturate a variable x with the limits x_min and x_max
   *
   * \param[in] x Value
   * \param[in] x_min Minimal value
   * \param[in] x_max Maximum value
   * \return Saturated value
   */
  inline double saturateValue(double x, double x_min, double x_max)
  {
    return std::min(std::max(x, x_min), x_max);
  }

  /*! Saturate the torque rate to not stress the motors
   *
   * \param[in] tau_d_calculated Calculated input torques
   * \param[out] tau_d_saturated Saturated torque values
   * \param[in] delta_tau_max
   */
  inline void saturateTorqueRate(const Eigen::VectorXd &tau_d_calculated, Eigen::VectorXd *tau_d_saturated, double delta_tau_max)
  {
    for (size_t i = 0; i < tau_d_calculated.size(); i++)
    {
      const double difference = tau_d_calculated[i] - tau_d_saturated->operator()(i);
      tau_d_saturated->operator()(i) += saturateValue(difference, -delta_tau_max, delta_tau_max);
    }
  }

  void MyController::initDesiredPose(const Eigen::Vector3d &position_d_target,
                                     const Eigen::Quaterniond &orientation_d_target)
  {
    this->setReferencePose(position_d_target, orientation_d_target);
    this->position_d_ = position_d_target;
    this->orientation_d_ = orientation_d_target;
  }

  void MyController::initNullspaceConfig(const Eigen::VectorXd &q_d_nullspace_target)
  {
    this->setNullspaceConfig(q_d_nullspace_target);
    this->q_d_nullspace_ = this->q_d_nullspace_target_;
  }

  void MyController::setNumberOfJoints(size_t n_joints)
  {
    if (n_joints < 0)
    {
      throw std::invalid_argument("Number of joints must be none negative");
    }
    this->n_joints_ = n_joints;
    this->q_ = Eigen::VectorXd::Zero(this->n_joints_);
    this->dq_ = Eigen::VectorXd::Zero(this->n_joints_);
    this->jacobian_ = Eigen::MatrixXd::Zero(6, this->n_joints_);
    this->q_d_nullspace_ = Eigen::VectorXd::Zero(this->n_joints_);
    this->q_d_nullspace_target_ = this->q_d_nullspace_;
    this->tau_c_ = Eigen::VectorXd::Zero(this->n_joints_);
    this->tau_m_ = Eigen::VectorXd::Zero(this->n_joints_);
  }

  void MyController::setStiffness(const Eigen::Matrix<double, 7, 1> &stiffness, bool auto_damping)
  {
    for (int i = 0; i < 6; i++)
    {
      // Set diagonal values of stiffness matrix
      if (stiffness(i) < 0.0)
      {
        assert(stiffness(i) >= 0 && "Stiffness values need to be positive.");
        this->cartesian_stiffness_target_(i, i) = 0.0;
      }
      else
      {
        this->cartesian_stiffness_target_(i, i) = stiffness(i);
      }
    }
    if (stiffness(6) < 0.0)
    {
      assert(stiffness(6) >= 0.0 && "Stiffness values need to be positive.");
      this->nullspace_stiffness_target_ = 0.0;
    }
    else
    {
      this->nullspace_stiffness_target_ = stiffness(6);
    }
    if (auto_damping)
    {
      this->applyDamping();
    }
  }

  void MyController::setStiffness(double t_x, double t_y, double t_z, double r_x, double r_y, double r_z,
                                  double n, bool auto_damping)
  {
    Eigen::Matrix<double, 7, 1> stiffness_vector(7);
    stiffness_vector << t_x, t_y, t_z, r_x, r_y, r_z, n;
    this->setStiffness(stiffness_vector, auto_damping);
  }

  void MyController::setStiffness(double t_x, double t_y, double t_z, double r_x, double r_y, double r_z, bool auto_damping)
  {
    Eigen::Matrix<double, 7, 1> stiffness_vector(7);
    stiffness_vector << t_x, t_y, t_z, r_x, r_y, r_z, this->nullspace_stiffness_target_;
    this->setStiffness(stiffness_vector, auto_damping);
  }

  void MyController::setDampingFactors(double d_x, double d_y, double d_z, double d_a, double d_b, double d_c,
                                       double d_n)
  {
    Eigen::Matrix<double, 7, 1> damping_new;
    damping_new << d_x, d_y, d_z, d_a, d_b, d_c, d_n;
    for (size_t i = 0; i < damping_new.size(); i++)
    {
      if (damping_new(i) < 0)
      {
        assert(damping_new(i) >= 0 && "Damping factor must not be negative.");
        damping_new(i) = this->damping_factors_(i);
      }
    }
    this->damping_factors_ = damping_new;
    this->applyDamping();
  }

  void MyController::applyDamping()
  {
    for (int i = 0; i < 6; i++)
    {
      assert(this->damping_factors_(i) >= 0.0 && "Damping values need to be positive.");
      this->cartesian_damping_target_(i, i) =
          this->damping_factors_(i) * this->dampingRule(this->cartesian_stiffness_target_(i, i));
    }
    assert(this->damping_factors_(6) >= 0.0 && "Damping values need to be positive.");
    this->nullspace_damping_target_ = this->damping_factors_(6) * this->dampingRule(this->nullspace_stiffness_target_);
  }

  void MyController::setReferencePose(const Eigen::Vector3d &position_d_target,
                                      const Eigen::Quaterniond &orientation_d_target)
  {
    this->position_d_target_ << position_d_target;
    this->orientation_d_target_.coeffs() << orientation_d_target.coeffs();
    this->orientation_d_target_.normalize();
  }

  void MyController::setNullspaceConfig(const Eigen::VectorXd &q_d_nullspace_target)
  {
    assert(q_d_nullspace_target.size() == this->n_joints_ && "Nullspace target needs to same size as n_joints_");
    this->q_d_nullspace_target_ << q_d_nullspace_target;
  }

  void MyController::setFiltering(double update_frequency, double filter_params_nullspace_config, double filter_params_stiffness,
                                  double filter_params_pose, double filter_params_wrench)
  {
    this->setUpdateFrequency(update_frequency);
    this->setFilterValue(filter_params_nullspace_config, &this->filter_params_nullspace_config_);
    this->setFilterValue(filter_params_stiffness, &this->filter_params_stiffness_);
    this->setFilterValue(filter_params_pose, &this->filter_params_pose_);
    this->setFilterValue(filter_params_wrench, &this->filter_params_wrench_);
  }

  void MyController::setMaxTorqueDelta(double d)
  {
    assert(d >= 0.0 && "Allowed torque change must be positive");
    this->delta_tau_max_ = d;
  }

  void MyController::setMaxTorqueDelta(double d, double update_frequency)
  {
    this->setMaxTorqueDelta(d);
    this->setUpdateFrequency(update_frequency);
  }

  void MyController::applyWrench(const Eigen::Matrix<double, 6, 1> &cartesian_wrench_target)
  {
    this->cartesian_wrench_target_ = cartesian_wrench_target;
  }

  Eigen::VectorXd MyController::calculateCommandedTorques(const Eigen::VectorXd &q,
                                                          const Eigen::VectorXd &dq,
                                                          const Eigen::Vector3d &position,
                                                          Eigen::Quaterniond orientation,
                                                          const Eigen::MatrixXd &jacobian)
  {
    // Update controller to the current robot state
    this->q_ = q;
    this->dq_ = dq;
    this->position_ << position;
    this->orientation_.coeffs() << orientation.coeffs();
    this->jacobian_ << jacobian;

    return this->calculateCommandedTorques();
  }

  Eigen::VectorXd MyController::calculateCommandedTorques()
  {
    // Perform a filtering step
    updateFilteredNullspaceConfig();
    updateFilteredStiffness();
    updateFilteredPose();
    updateFilteredWrench();
    // Compute error term
    this->error_.head(3) << this->position_ - this->position_d_;
    this->error_.tail(3) << calculateOrientationError(this->orientation_d_, this->orientation_);
    // Kinematic pseuoinverse
    Eigen::MatrixXd jacobian_transpose_pinv;
    pseudoInverse((this->jacobian_).block<6, 7>(0, 0).transpose(), &jacobian_transpose_pinv);
    Eigen::VectorXd tau_task(this->n_joints_), tau_nullspace(this->n_joints_), tau_ext(this->n_joints_);

    std::cout <<"jacobian"<< (this->jacobian_).block<6, 7>(0, 0) << std::endl;

    // Torque calculated for Cartesian impedance control with respect to a Cartesian pose reference in the end, in the frame of the EE of the robot.
    tau_task << (this->jacobian_).block<6, 7>(0, 0).transpose() * (-this->cartesian_stiffness_ * this->error_ - this->cartesian_damping_ * ((this->jacobian_).block<6, 7>(0, 0) * this->dq_));

    // Torque for joint impedance control with respect to a desired configuration and projected in the null-space of the robot's Jacobian, so it should not affect the Cartesian motion of the robot's end-effector.
    tau_nullspace << (Eigen::MatrixXd::Identity(7, 7) - (this->jacobian_).block<6, 7>(0, 0).transpose() * jacobian_transpose_pinv) *
                         (this->nullspace_stiffness_ * (this->q_d_nullspace_ - this->q_) - this->nullspace_damping_ * this->dq_);

    // Torque to achieve the desired external force command in the frame of the EE of the robot.
    tau_ext = (this->jacobian_).block<6, 7>(0, 0).transpose() * this->cartesian_wrench_;

    // Torque commanded to the joints of the robot is composed by the superposition of these three joint-torque signals:
    Eigen::VectorXd tau_d = tau_task + tau_nullspace + tau_ext;
    //saturateTorqueRate(tau_d, &this->tau_c_, this->delta_tau_max_);
    /*
    std::cout <<"tau_c\n"<< std::endl;
    std::cout <<tau_c_<< std::endl;
    std::cout <<"tau_task\n"<< std::endl;
    std::cout <<tau_task<< std::endl;
    std::cout <<"tau_nullspace\n"<< std::endl;
    std::cout <<tau_nullspace<< std::endl;
    std::cout <<"tau_ext\n"<< std::endl;
    std::cout <<tau_ext<< std::endl;
    std::cout <<"tau_d"<< std::endl;
    std::cout <<tau_d<< std::endl;
    std::cout <<"tau_m"<< std::endl;
    std::cout <<this->tau_m_<< std::endl;
    std::cout <<"calculate torques end"<< std::endl;
    */
    Eigen::VectorXd tau_test(7); 
    tau_test << 0, 0, 0, 0,  0, 0, 0;
    this->tau_c_ = tau_test;
    return this->tau_c_;
  }

  // Get the state of the robot.Updates when "calculateCommandedTorques" is called
  void MyController::getState(Eigen::VectorXd *q, Eigen::VectorXd *dq, Eigen::Vector3d *position,
                              Eigen::Quaterniond *orientation, Eigen::Vector3d *position_d,
                              Eigen::Quaterniond *orientation_d,
                              Eigen::Matrix<double, 6, 6> *cartesian_stiffness,
                              double *nullspace_stiffness, Eigen::VectorXd *q_d_nullspace,
                              Eigen::Matrix<double, 6, 6> *cartesian_damping) const
  {
    *q << this->q_;
    *dq << this->dq_;
    *position << this->position_;
    orientation->coeffs() << this->orientation_.coeffs();
    this->getState(position_d, orientation_d, cartesian_stiffness, nullspace_stiffness, q_d_nullspace, cartesian_damping);
  }

  void MyController::getState(Eigen::Vector3d *position_d, Eigen::Quaterniond *orientation_d,
                              Eigen::Matrix<double, 6, 6> *cartesian_stiffness,
                              double *nullspace_stiffness, Eigen::VectorXd *q_d_nullspace,
                              Eigen::Matrix<double, 6, 6> *cartesian_damping) const
  {
    *position_d = this->position_d_;
    orientation_d->coeffs() << this->orientation_d_.coeffs();
    *cartesian_stiffness = this->cartesian_stiffness_;
    *nullspace_stiffness = this->nullspace_stiffness_;
    *q_d_nullspace = this->q_d_nullspace_;
    *cartesian_damping << this->cartesian_damping_;
  }

  Eigen::VectorXd MyController::getLastCommands() const
  {
    return this->tau_c_;
  }

  Eigen::Matrix<double, 6, 1> MyController::getAppliedWrench() const
  {
    return this->cartesian_wrench_;
  }

  Eigen::Matrix<double, 6, 1> MyController::getPoseError() const
  {
    return this->error_;
  }

  double MyController::dampingRule(double stiffness) const
  {
    return 2 * sqrt(stiffness);
  }

  void MyController::setUpdateFrequency(double freq)
  {
    assert(freq >= 0.0 && "Update frequency needs to be greater or equal to zero");
    this->update_frequency_ = std::max(freq, 0.0);
  }

  void MyController::setFilterValue(double val, double *saved_val)
  {
    assert(val > 0 && val <= 1.0 && "Filter params need to be between 0 and 1.");
    *saved_val = saturateValue(val, 0.0000001, 1.0);
  }

  void MyController::updateFilteredNullspaceConfig()
  {
    if (this->filter_params_nullspace_config_ == 1.0)
    {
      this->q_d_nullspace_ = this->q_d_nullspace_target_;
    }
    else
    {
      const double step = filterStep(this->update_frequency_, this->filter_params_nullspace_config_);
      this->q_d_nullspace_ = filteredUpdate(this->q_d_nullspace_target_, this->q_d_nullspace_, step);
    }
  }

  void MyController::updateFilteredStiffness()
  {
    if (this->filter_params_stiffness_ == 1.0)
    {
      this->cartesian_stiffness_ = this->cartesian_stiffness_target_;
      this->cartesian_damping_ = this->cartesian_damping_target_;
      this->nullspace_stiffness_ = this->nullspace_stiffness_target_;
      this->q_d_nullspace_ = this->q_d_nullspace_target_;
      this->nullspace_damping_ = this->nullspace_damping_target_;
    }
    else
    {
      const double step = filterStep(this->update_frequency_, this->filter_params_stiffness_);

      this->cartesian_stiffness_ = filteredUpdate(this->cartesian_stiffness_target_, this->cartesian_stiffness_, step);
      this->cartesian_damping_ = filteredUpdate(this->cartesian_damping_target_, this->cartesian_damping_, step);
      this->nullspace_stiffness_ = filteredUpdate(this->nullspace_stiffness_target_, this->nullspace_stiffness_, step);
      this->nullspace_damping_ = filteredUpdate(this->nullspace_damping_target_, this->nullspace_damping_, step);
    }
  }

  void MyController::updateFilteredPose()
  {
    if (this->filter_params_pose_ == 1.0)
    {
      position_d_ << position_d_target_;
      orientation_d_.coeffs() << orientation_d_target_.coeffs();
    }
    else
    {
      const double step = filterStep(this->update_frequency_, this->filter_params_pose_);

      this->position_d_ = filteredUpdate(this->position_d_target_, this->position_d_, step);
      this->orientation_d_ = this->orientation_d_.slerp(step, this->orientation_d_target_);
    }
  }

  void MyController::updateFilteredWrench()
  {
    if (this->filter_params_wrench_ == 1.0)
    {
      this->cartesian_wrench_ = this->cartesian_wrench_target_;
    }
    else
    {
      const double step = filterStep(this->update_frequency_, this->filter_params_wrench_);
      this->cartesian_wrench_ = filteredUpdate(this->cartesian_wrench_target_, this->cartesian_wrench_, step);
    }
  }

  controller_interface::CallbackReturn MyController::on_init()
  {
    const std::string urdf_filename = "/home/pavi/ros2_ws/src/my_controller/controller/urdf/panda.urdf";
    pinocchio::urdf::buildModel(urdf_filename, this->model_);
    pinocchio::Data data(this->model_);
    this->data_ = data;
    // should have error handling
    joint_names_ = auto_declare<std::vector<std::string>>("joints", joint_names_);

    command_interface_types_ = auto_declare<std::vector<std::string>>("command_interfaces", command_interface_types_);
   
    state_interface_types_ = auto_declare<std::vector<std::string>>("state_interfaces", state_interface_types_);
    
    point_interp_.positions.assign(joint_names_.size(), 0);
    point_interp_.velocities.assign(joint_names_.size(), 0);
    point_interp_.effort.assign(joint_names_.size(), 0); //Why?
    setNumberOfJoints(joint_names_.size());
    return CallbackReturn::SUCCESS;
  }

  controller_interface::InterfaceConfiguration MyController::command_interface_configuration()
      const
  {
    controller_interface::InterfaceConfiguration conf = {config_type::INDIVIDUAL, {}};

    conf.names.reserve(joint_names_.size() * command_interface_types_.size());
    for (const auto &joint_name : joint_names_)
    {
      for (const auto &interface_type : command_interface_types_)
      {
        conf.names.push_back(joint_name + "/" + interface_type);
      }
    }
    return conf;
  }

  controller_interface::InterfaceConfiguration MyController::state_interface_configuration() const
  {
    controller_interface::InterfaceConfiguration conf = {config_type::INDIVIDUAL, {}};

    conf.names.reserve(joint_names_.size() * state_interface_types_.size());
    for (const auto &joint_name : joint_names_)
    {
      for (const auto &interface_type : state_interface_types_)
      {
        conf.names.push_back(joint_name + "/" + interface_type);
      }
    }
    return conf;
  }

  controller_interface::CallbackReturn MyController::on_configure(const rclcpp_lifecycle::State &)
  {
    auto callback =
        [this](const std::shared_ptr<trajectory_msgs::msg::JointTrajectory> traj_msg) -> void
    {
      traj_msg_external_point_ptr_.writeFromNonRT(traj_msg);
      new_msg_ = true;
    };
    return CallbackReturn::SUCCESS;
  }

  controller_interface::CallbackReturn MyController::on_activate(const rclcpp_lifecycle::State &)
  {
    // clear out vectors in case of restart
    joint_effort_command_interface_.clear();
    joint_position_state_interface_.clear();
    joint_velocity_state_interface_.clear();

    

    // assign command interfaces
    for (auto &interface : command_interfaces_)
    {
      command_interface_map_[interface.get_interface_name()]->push_back(interface);
    }
    
    // assign state interfaces
    for (auto &interface : state_interfaces_)
    {
      state_interface_map_[interface.get_interface_name()]->push_back(interface);
    }
    
    trajectory_service_ =
        get_node()->create_service<my_controller_interface::srv::MyController>(
            "~/joint_trajectory", std::bind(&MyController::initTrajectory, this, std::placeholders::_1, std::placeholders::_2));

    return CallbackReturn::SUCCESS;
  }

  void interpolate_point(
      const trajectory_msgs::msg::JointTrajectoryPoint &point_1,
      const trajectory_msgs::msg::JointTrajectoryPoint &point_2,
      trajectory_msgs::msg::JointTrajectoryPoint &point_interp, double delta)
  {
    for (size_t i = 0; i < point_1.positions.size(); i++)
    {
      point_interp.positions[i] = delta * point_2.positions[i] + (1.0 - delta) * point_2.positions[i];
    }
    for (size_t i = 0; i < point_1.positions.size(); i++)
    {
      point_interp.velocities[i] =
          delta * point_2.velocities[i] + (1.0 - delta) * point_2.velocities[i];
    }
  }

  void interpolate_trajectory_point(
      const trajectory_msgs::msg::JointTrajectory &traj_msg, const rclcpp::Duration &cur_time,
      trajectory_msgs::msg::JointTrajectoryPoint &point_interp)
  {
    double traj_len = traj_msg.points.size();
    auto last_time = traj_msg.points[traj_len - 1].time_from_start;
    double total_time = last_time.sec + last_time.nanosec * 1E-9;

    size_t ind = cur_time.seconds() * (traj_len / total_time);
    ind = std::min(static_cast<double>(ind), traj_len - 2);
    double delta = cur_time.seconds() - ind * (total_time / traj_len);
    interpolate_point(traj_msg.points[ind], traj_msg.points[ind + 1], point_interp, delta);
  }

  controller_interface::return_type MyController::update(
      const rclcpp::Time & /*time*/, const rclcpp::Duration & /*period*/)
  {
    std::cout <<"update start"<< std::endl;
    if (this->traj_running_)
    {
      trajUpdate();
    }

    updateState();
    // Apply control law in base library
    //this->calculateCommandedTorques();
    for (size_t i = 0; i < joint_effort_command_interface_.size(); i++)
    {
      joint_effort_command_interface_[i].get().set_value(0.5);
    }
    // auto timestamp = std::chrono::high_resolution_clock::now();
    // std::cout << "Command sent at: " << std::chrono::duration_cast<std::chrono::milliseconds>(timestamp.time_since_epoch()).count() << " milliseconds\n";
    // std::cout <<"update end"<< std::endl;
    return controller_interface::return_type::OK;
  }

  controller_interface::CallbackReturn MyController::on_deactivate(const rclcpp_lifecycle::State &)
  {
    release_interfaces();

    return CallbackReturn::SUCCESS;
  }

  controller_interface::CallbackReturn MyController::on_cleanup(const rclcpp_lifecycle::State &)
  {
    return CallbackReturn::SUCCESS;
  }

  controller_interface::CallbackReturn MyController::on_error(const rclcpp_lifecycle::State &)
  {
    return CallbackReturn::SUCCESS;
  }

  controller_interface::CallbackReturn MyController::on_shutdown(const rclcpp_lifecycle::State &)
  {
    return CallbackReturn::SUCCESS;
  }

  void MyController::trajUpdate()
  {

    Eigen::Vector3d position_d_error = (this->position_d_target_) - (this->position_d_);
    if (position_d_error.norm() <= 0.01)
    {
      this->traj_index_++;
      // Get end effector pose
      Eigen::VectorXd q = Eigen::VectorXd::Map(trajectory_.points.at(this->traj_index_).positions.data(),
                                               trajectory_.points.at(this->traj_index_).positions.size());
      getFk(q, &this->position_d_target_, &this->orientation_d_target_);
      this->setNullspaceConfig(q);
    }
    if (this->traj_index_ > this->trajectory_.points.size())
    {
      this->traj_running_ = false;
    }
    // Update end-effector pose and nullspace

    // if (ros::Time::now() > (this->traj_start_ + this->traj_duration_))
    // {
    //   ROS_INFO_STREAM("Finished executing trajectory.");
    //   if (this->traj_as_->isActive())
    //   {
    //     this->traj_as_->setSucceeded();
    //   }
    //   this->traj_running_ = false;
    // }
  }

  bool MyController::getFk(const Eigen::VectorXd &q, Eigen::Vector3d *position,
                           Eigen::Quaterniond *orientation)
  {

    using namespace pinocchio;

    //Baustelle diese, mies hässlich (wie deine mom)
    Eigen::VectorXd q_ext = Eigen::VectorXd::Zero(9);
    
    for(unsigned int i = 0; i<7;++i){
      q_ext[i] = q[i];
    }
    // // Create data required by the algorithms
    
    // // Perform the forward kinematics over the kinematic tree
    pinocchio::forwardKinematics(model_, data_, q_ext);
    //pinocchio::updateFramePlacements(model_, data_);
    
    // *position = data.oMf[frame_id].translation();
    // *orientation = Eigen::Quaterniond(data.oMf[frame_id].rotation());
    return true;
  }

  void MyController::initTrajectory(const std::shared_ptr<my_controller_interface::srv::MyController::Request> request,
                                    std::shared_ptr<my_controller_interface::srv::MyController::Response> response)
  {
    // error handeling needed, maybe transfer to action server
    trajectory_ = request->trajectory;
    response->success = true;
  }

  void MyController::updateState()
  {
    for (size_t i = 0; i < joint_names_.size(); i++)
    { 
      
      q_[i] = joint_position_state_interface_.at(i).get().get_value();
      dq_[i] = joint_velocity_state_interface_.at(i).get().get_value();
      tau_m_[i] = joint_effort_state_interface_.at(i).get().get_value();  
    }
    // std::cout<<this->q_<<std::endl;

    getJacobian();
    //getFk(this->q_, &this->position_, &this->orientation_);
  }

  bool MyController::getJacobian()
  {
    using namespace pinocchio;

    //forwardKinematics(model_, data_, q); //Why?
    jacobian_ = computeJointJacobians(model_, data_);
    std::cout<<this->jacobian_<<std::endl;
    return true;
  }

} // namespace ros2_control_demo_example_7

#include "pluginlib/class_list_macros.hpp"

PLUGINLIB_EXPORT_CLASS(
    my_controller::MyController, controller_interface::ControllerInterface)