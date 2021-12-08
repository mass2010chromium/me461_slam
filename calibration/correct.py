        map_thresh = map_img[:, :, 0] > 70
        centered_lines = []
        for x1, y1, x2, y2 in scaled_lines:
            centered_lines.append([x1 - pose_px_x, y1 - pose_px_y,
                                   x2 - pose_px_x, y2 - pose_px_y])
        confidences = []
        #deltas = [-0.02, -0.015, -0.01, -0.005, 0, 0.005, 0.01, 0.015, 0.02]
        #deltas = [-0.04, -0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03, 0.04]
        #deltas = [-0.08, -0.06, -0.04, -0.02, 0, 0.02, 0.04, 0.06, 0.08]
        #deltas = [-0.08, -0.07, -0.06, -0.05, -0.04, -0.03, -0.02, -0.01,
        #          0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
        #deltas = np.array([-0.08, -0.07, -0.06, -0.05, -0.04, -0.03, -0.02, -0.01,
        #          0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]) * 4
        deltas = np.linspace(-0.4, 0.4, 41)
        for delta_angle in deltas:
            rot_mat = np.array([[np.cos(delta_angle), np.sin(delta_angle)],
                                [-np.sin(delta_angle), np.cos(delta_angle)]])
            transform_lines = [np.array([*(rot_mat @ line[:2] + pose_px),
                                         *(rot_mat @ line[2:] + pose_px)],
                                         dtype=np.int32) for line in centered_lines]
            lines_only = zero_mask.copy()
            plot_lines(lines_only, transform_lines, 1)
            lines_only *= circle_mask
            line_px = np.where(lines_only == 1)
            max_score = len(line_px[0])
            score = np.sum(map_thresh[line_px])
            confidences.append(score / (max_score))
        weight = max(confidences)
        if weight > 0.4:
            avg = np.mean(confidences)
            print("confidence max, mean:", weight, avg)
            weight -= avg
            if weight > 0.4:
                confidences = np.array(confidences) - avg
                indices = confidences > 0.4
                total = np.sum(confidences[indices])
                net_delta = vo.dot(confidences[indices], deltas[indices]) / total
                if net_delta != 0:
                    print("Drift detected", confidences)
                print("guess:", net_delta)
                estimated_rot_err -= net_delta
                correct_pose = pose['heading'] + estimated_rot_err
                pose['heading'] = (1-weight) * pose['heading'] + weight*correct_pose
        else:
            print("bad data")
