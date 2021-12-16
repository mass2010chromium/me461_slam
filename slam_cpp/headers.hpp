#define BOOST_SPIRIT_THREADSAFE
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#ifdef DEBUG
#include <debug/vector>
using __gnu_debug::vector;
#else
#include <vector>
using std::vector;
#endif

#include <simple-web-server/client_http.hpp>

#include "types.h"
#include <motionlib/vectorops.h>
#include <motionlib/so3.h>
#include <motionlib/se3.h>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/calib3d.hpp>
