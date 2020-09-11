
**test_start_pose.npy**: Array with dim [number of samples, 3]. The x, y, and theta of the start object pose.

**test_contour_pt.npy**: Array with dim [number of samples, 3]. The x, y, and z of the point of contact between the robot end effector and the object.

**test_ee_vel.npy**: Array with dim [number of samples, 2]. The x, y velocity of the robot end effector at point of contact.

**test_end_pose.npy**: Array with dim [number of samples, 3]. The x, y, and theta of the end object pose.

**feed_forward_displacements.npy**: Array with dim [number of samples, 1]. Position displacement of the object calculated by `np.linalg.norm(start object position - end object position)`.

**feedforward_position_pred.npy**: Array with dim [number of samples, 2]. x, y predictions of test data set by feedforward model (`position_model.pt`).

**inertia_matrix.npy**: Array with dim [number of samples, 6, 6]. Last inertia matrix associated with test poke.
