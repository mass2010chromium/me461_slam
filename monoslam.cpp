#include <VNL/Algo/cholesky.h>
#include <VNL/Algo/determinant.h>

#include <Scene/control_general.h>
#include <MonoSLAM/nonoverlappingregion.h>

#include "monoslam.h"

MonoSLAM2::MonoSLAM2(unsigned int num_features_select,
        unsigned int num_features_visible,
        unsigned int max_features_init,
        double min_lambda,
        double max_lambda,
        unsigned int num_particles,
        double stdev_depth_ratio,
        unsigned int min_particles,
        double prune_threshold,
        unsigned int init_fail_threshold) {
    // TODO
}

MonoSLAM2::GoOneStep(const cv::Mat& image, double dt, bool do_map) {

}
