#pragma once

/*
 * ME 461 edition monoslam
 * so worse and like 10y out of date
 */

#include <vector>
#include <VNL/vector.h>
#include <Scene/models_base.h>
#include <Scene/kalman.h>

#include <opencv2/opencv.hpp>

class MonoSLAM2
{
    public:
        MonoSLAM2(unsigned int num_features_select,
                  unsigned int num_features_visible,
                  unsigned int max_features_init,
                  double min_lambda,
                  double max_lambda,
                  unsigned int num_particles,
                  double stdev_depth_ratio,
                  unsigned int min_particles,
                  double prune_threshold,
                  unsigned int init_fail_threshold);

        virtual ~MonoSLAM2() {};

    public:
        bool GoOneStep(const cv::Mat &image, double dt, bool do_map);

        unsigned int CAMERA_WIDTH;
        unsigned int CAMERA_HEIGHT;

        unsigned int NUM_FEATURES_SELECT;
        unsigned int NUM_FEATURES_VISIBLE;
        unsigned int MAX_FEATURES_INIT;

        double MIN_LAMBDA;
        double MAX_LAMBDA;
        unsigned int NUM_PARTICLES;
        double STDEV_DEPTH_RATIO;
        unsigned int MIN_PARTICLES;
        double PRUNE_THRESHOLD;
        unsigned int INIT_FAIL_THRESHOLD;
        
        int init_search_ustart;
        int init_search_vstart;
        int init_search_ufinish;
        int init_search_ufinish;
        bool init_search_region_defined;
        
        unsigned int num_visible_features;
        unsigned int num_matched_features;
        VNL::VectorFixed<3, double> velocity;
};
