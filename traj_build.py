import numpy as np
import math


def build_trajectories(centres, tc, tl, mfs, mft):

    # Set constants
    threshold_c = tc
    threshold_l = tl
    max_frames_seed = mfs
    max_frame_trajectory = mft

    # Empty arrays to hold seeds and trajectories
    seeds = []
    trajectories = []
    old_trajectories = []

    # Loop through every point and try match it to a existing seed or trajectory - if no match then create new seed
    for i in range(len(centres)):
        f = centres[i][0][2]
        print(f)                    # Current frame number
        if f == 92:
            g = 1
            pass
        new_seeds = []          # Empty array that will store all unmatched seeds from this frame

        # Remove unmatched seeds that occurred more than [max_frames] frames ago
        remove_old = []
        for s, seed1 in enumerate(seeds):
            if f - seed1[2] > max_frames_seed:
                remove_old.append(s)

        old_seed_count = 0
        for r in remove_old:
            del seeds[r-old_seed_count]
            old_seed_count +=1  # Every time a seed is removed the index of the other seeds reduces by 1

        # Move older trajectories into a new array if there last point is older than [max_frames] frames
        remove_traj = []
        for t, traj1 in enumerate(trajectories):
            if f-traj1[-1][2] > max_frame_trajectory:
                old_trajectories.append(traj1)
                remove_traj.append(t)

        old_traj_count = 0
        for a in remove_traj:
            del trajectories[a-old_traj_count]
            old_traj_count += 1 # Every time a traj is removed the index of the other trajectories reduces by 1

        # Empty arrays to store points that will be added to the relevant trajectories
        index_traj = []
        cen_traj = []
        dist_score = []

        # Loop through all points in current frame and try match them to existing seeds or trajectories
        for count_cen, cen in enumerate(centres[i]):
            traj_match = False
            seed_match = False
            removes = []

            # Loop through trajectories to find match
            for count_t, traj in enumerate(trajectories):
                inv_len = (-1) * len(traj)  # the invert length enables one to iterate backwards through the array
                M, C = 0, 0                 # Gradient and intercept of line (set initially to zero)

                # Loop backwards through the array (starting at the second last point) creating a line with the last point and the second last (non-duplicate) point
                Vert = False
                Hor = False
                Normal = False
                for k in range(-2, inv_len - 1, -1):
                    run = traj[-1][0] - traj[k][0]
                    rise = traj[-1][1] - traj[k][1]

                    if (run == 0) and (rise == 0):   # duplicate point (no line)
                        continue
                    elif run == 0:                 # vertical line
                        Vert = True
                        break
                    elif rise == 0:                 # horizontal line
                        Hor = True
                        break
                    else:                           # normal line
                        M = rise / run  # M = (y1-y0)/(x1-x0)
                        C = traj[-1][1] - M * traj[-1][0]  # C = y-Mx
                        Normal = True
                        break

                if Normal:
                    yline = M * (cen[0]) + C
                    xline = (cen[1] - C) / M

                    xl_x = abs(xline - cen[0])
                    yl_y = abs(yline - cen[1])

                    if xl_x == 0:  # if either xl_x or yl_y is 0 then the other one will be too and it means the point is exactly on the line
                        dline = 0

                    else:
                        theta = math.atan(yl_y / xl_x)

                        dline = yl_y * math.cos(theta)

                elif Hor:
                    yline = traj[-1][1]
                    dline = abs(yline - cen[1])

                elif Vert:
                    xline = traj[-1][0]
                    dline = abs(xline - cen[0])

                else:
                    break

                dx = np.abs(cen[0] - traj[-1][0])
                dy = np.abs(cen[1] - traj[-1][1])
                dr = np.sqrt(dx ** 2 + dy ** 2)

                if (dline < threshold_l) & (dr < threshold_c):
                    index_traj.append(count_t)  # the index of the trajectory that the point should be inserted into
                    cen_traj.append(cen)       # the actual point to be inserted
                    dist_score.append([dline, dr])
                    traj_match = True

            if not traj_match:
                # Doesnt match trajectory so check if it matches an existing seed
                for count_s, seed in enumerate(seeds):
                    dx = np.abs(cen[0] - seed[0])
                    dy = np.abs(cen[1] - seed[1])
                    dr = np.sqrt(dx ** 2 + dy ** 2) # The distance between the two points

                    # If the distance is less than the threshold and its not the exact same point (dr == 0) then create a trajectory form the two seeds
                    if (dr < threshold_c) & (dr > 0):
                        trajectories.append([seed, cen])
                        removes.append(count_s)
                        seed_match = True

                # Remove the seed that the point was close to because that seed is now part of a trajectory
                if seed_match:
                    match_count = 0
                    for x in removes:
                        del seeds[x-match_count]
                        match_count += 1

            # If no match is found the point becomes a new seed
            if (not traj_match) & (not seed_match):
                new_seeds.append(cen)

        # Add all points that matched to there corresponding trajectories

        # Firstly, only one point can be added to any one trajectory so in the case that there are two points; the one with the lower score is removed
        remove_point = []
        for a in range(len(index_traj)):
            out = False
            rb = 0
            for b in range(a + 1, len(index_traj)):
                if index_traj[a] == index_traj[b-rb]:
                    dl1 = dist_score[a][0]
                    dr1 = dist_score[a][1]
                    sc1 = cen_traj[a][3]
                    dl2 = dist_score[b-rb][0]
                    dr2 = dist_score[b-rb][1]
                    sc2 = cen_traj[b-rb][3]
                    if sc1 > sc2:
                        del index_traj[b-rb]
                        del cen_traj[b-rb]
                        del dist_score[b-rb]
                        rb += 1
                    else:
                        remove_point.append(a)
                        out = True
                        break
            if out:
                continue

        num_removes = 0
        for point in remove_point:
            del index_traj[point - num_removes]
            del cen_traj[point - num_removes]
            del dist_score[point-num_removes]
            num_removes += 1

        for count_cen, ind in enumerate(index_traj):
            trajectories[ind].append(cen_traj[count_cen])


        for new_seed in new_seeds:
            seeds.append(new_seed)

    for traj2 in trajectories:
        old_trajectories.append(traj2)

    return old_trajectories
