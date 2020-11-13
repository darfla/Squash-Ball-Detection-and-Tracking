from traj_build import build_trajectories
from traj_correction import build_features
from traj_correction import correct_trajectories
from traj_interpolation import*
import matplotlib.pyplot as plt


def track_ball(centres, original, bs_orig, model_obj, scaler_obj):
    trajectories1 = build_trajectories(centres.copy(), tc=60, tl=20, mfs=5, mft=5)

    traj_features_b = build_features(trajectories1.copy())
    print(traj_features_b)
    for fj in trajectories1:
        print(fj)

    trajectories_correct, traj_features_c = correct_trajectories(trajectories1.copy(), traj_features_b.copy(),
                                                                 max_score_thresh=0.7, len_thresh=2.5)


    trajectories_intra = intrapolate_traj(trajectories_correct.copy(), original, bs_orig,
                                          model_obj, scaler_obj, tc=40, tl=20)

    traj_features_i = build_features(trajectories_intra.copy())


    x, y, f = plot_traj(centres)
    xt, yt, ft = plot_traj(trajectories1.copy())
    xc, yc, fc = plot_traj(trajectories_correct.copy())
    xi, yi, fi = plot_traj(trajectories_intra.copy())

    # print(xc)
    # print(fc)
    # print(xi)
    # print(fi)
    print(traj_features_i)
    for tj in trajectories_intra:
        print(tj)
    xmax = 1280
    ymax = 720

    f3, (ax4, ax5) = plt.subplots(1, 2)
    ax4.scatter(f, y)
    ax4.set_title('Y vs Frame')
    ax4.set_ylim(ymax, 0)
    ax4.set_xlabel('F')
    ax4.set_ylabel('Y')
    ax5.scatter(f, x)
    ax5.set_title('X vs Frame')
    ax5.set_ylim(0, xmax)
    ax5.set_xlabel('F')
    ax5.set_ylabel('X')

    f4, (ax6, ax7) = plt.subplots(1, 2)
    ax6.scatter(ft, yt)
    ax6.set_title('Yt vs Frame')
    ax6.set_ylim(ymax, 0)
    ax6.set_xlabel('F')
    ax6.set_ylabel('Yt')
    ax7.scatter(ft, xt)
    ax7.set_title('Xt vs Frame')
    ax7.set_ylim(0, xmax)
    ax7.set_xlabel('F')
    ax7.set_ylabel('Xt')

    f5, (ax8, ax9) = plt.subplots(1, 2)
    ax8.scatter(fc, yc)
    ax8.set_title('Ytc vs Frame')
    ax8.set_ylim(ymax, 0)
    ax8.set_xlabel('F')
    ax8.set_ylabel('Ytc')
    ax9.scatter(fc, xc)
    ax9.set_title('Xtc vs Frame')
    ax9.set_ylim(0, xmax)
    ax9.set_xlabel('F')
    ax9.set_ylabel('Xtc')
    #
    # f6, (ax10, ax11) = plt.subplots(1, 2)
    # ax10.scatter(fi, yi)
    # ax10.set_title('Yi vs Frame')
    # ax10.set_ylim(ymax, 0)
    # ax10.set_xlabel('F')
    # ax10.set_ylabel('Yi')
    # ax11.scatter(fi, xi)
    # ax11.set_title('Xi vs Frame')
    # ax11.set_ylim(0, xmax)
    # ax11.set_xlabel('F')
    # ax11.set_ylabel('Xi')

    plt.show()

    return sort_data(xi, yi, fi)

