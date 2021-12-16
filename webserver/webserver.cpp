#include <simple-web-server/server_http.hpp>
#include <future>
#include <math.h>
#include <memory>

#include <unistd.h>
#include <sys/mman.h>

// Added for the json-example
#define BOOST_SPIRIT_THREADSAFE
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include <boost/lockfree/queue.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

// Added for the default_resource example
#include <algorithm>
#include <boost/filesystem.hpp>
#include <fstream>
#include <vector>
#ifdef HAVE_OPENSSL
#include "crypto.hpp"
#endif

#include <stdio.h>
#include "serial_dev.h"

#include <mutex>
std::mutex slam_pose_lock;

using namespace std;
// Added for the json-example:
using namespace boost::property_tree;

using HttpServer = SimpleWeb::Server<SimpleWeb::HTTP>;

char* map_file(std::string filename, int open_flags, size_t map_size) {
    cout << "Mapping file " << filename << endl;
    int mmap_file = open(filename.c_str(), open_flags, 0777);
    if (mmap_file == -1) {
        perror("open mmap file failure");
        return NULL;
    }
    ftruncate(mmap_file, map_size);
    char* ret = (char*) mmap(NULL, map_size, PROT_READ | PROT_WRITE, MAP_SHARED_VALIDATE, mmap_file, 0);
    if (ret == NULL) {
        perror("mmap failure");
        return NULL;
    }
    return ret;
}


const float FEET_TO_METER = 0.3048;
const float METER_TO_FEET = 3.2808399;

struct RobotInfo {
    float x;
    float y;
    float heading;
    float v;
    float w;
};
typedef struct RobotInfo RobotInfo;

RobotInfo robot_info;
RobotInfo slam_info;

struct RobotCommand {
  RobotCommand() : cmd_v(0), cmd_w(0) {}
  RobotCommand(float v, float w) : cmd_v(v), cmd_w(w) {}
  RobotCommand(float x, float y, float t) : cmd_x(x), cmd_y(y), cmd_heading(t) {}
  union {
    float cmd_v;
    float cmd_x;
  };
  union {
    float cmd_w;
    float cmd_y;
  };
  float cmd_heading;
};
typedef struct RobotCommand RobotCommand;

boost::lockfree::queue<RobotCommand> command_queue(128);
boost::lockfree::queue<RobotCommand> target_queue(16);
RobotCommand current_target(0, 0, 0);

// 0: idle
// 1: planning
// 2: failed
int planner_status = 0;

std::mutex image_lock;
std::shared_ptr<vector<uchar>> image_data;

std::mutex map_lock;
std::shared_ptr<vector<uchar>> map_data;

// lol single threaded server go brr
int main() {
  // HTTP-server at port 8080 using 1 thread
  // Unless you do more heavy non-threaded processing in the resources,
  // 1 thread is usually faster than several threads
  HttpServer server;
  server.config.port = 8080;

  // We're gonna "serve" camera data through an mmap'd file.
  cv::VideoCapture camera(0);
  camera.set(cv::CAP_PROP_BUFFERSIZE, 1);

  // GET-example for the path /info
  // Responds with request-information
  server.resource["^/info$"]["GET"] = [](shared_ptr<HttpServer::Response> response, shared_ptr<HttpServer::Request> request) {
    stringstream stream;
    stream << "<h1>Request from " << request->remote_endpoint().address().to_string() << ":" << request->remote_endpoint().port() << "</h1>";

    stream << request->method << " " << request->path << " HTTP/" << request->http_version;

    stream << "<h2>Query Fields</h2>";
    auto query_fields = request->parse_query_string();
    for(auto &field : query_fields)
      stream << field.first << ": " << field.second << "<br>";

    stream << "<h2>Header Fields</h2>";
    for(auto &field : request->header)
      stream << field.first << ": " << field.second << "<br>";

    response->write(stream);
  };
  
  server.resource["^/target$"]["GET"] = [](shared_ptr<HttpServer::Response> response, shared_ptr<HttpServer::Request> /*request*/) {
    stringstream stream;
    bool new_request = target_queue.pop(current_target);
    stream << "{\"x\":" << current_target.cmd_x
           << ",\"y\":" << current_target.cmd_y
           << ",\"heading\":" << current_target.cmd_heading
           << ",\"new_request\":" << new_request << "}";
    response->write(stream);
  };

  server.resource["^/target$"]["POST"] = [](shared_ptr<HttpServer::Response> response, shared_ptr<HttpServer::Request> request) {
    try {
      ptree pt;
      read_json(request->content, pt);
      float cmd_x = pt.get<float>("x");
      float cmd_y = pt.get<float>("y");
      float cmd_heading = pt.get<float>("t");
      target_queue.push(RobotCommand(cmd_x, cmd_y, cmd_heading));
      response->write("POSE Command recieved");
    }
    catch (const exception& e) {
      response->write(SimpleWeb::StatusCode::client_error_bad_request, e.what());
    }
  };

  server.resource["^/planner_status$"]["GET"] = [](shared_ptr<HttpServer::Response> response, shared_ptr<HttpServer::Request> /*request*/) {
    stringstream stream;
    stream << "{\"status\":" << planner_status << "}";
    response->write(stream);
  };

  server.resource["^/planner_status$"]["POST"] = [](shared_ptr<HttpServer::Response> response, shared_ptr<HttpServer::Request> request) {
    try {
      ptree pt;
      read_json(request->content, pt);
      planner_status = pt.get<int>("status");
      response->write("STATUS Command recieved");
    }
    catch (const exception& e) {
      response->write(SimpleWeb::StatusCode::client_error_bad_request, e.what());
    }
  };

  server.resource["^/pose_slam$"]["GET"] = [](shared_ptr<HttpServer::Response> response, shared_ptr<HttpServer::Request> /*request*/) {
    stringstream stream;
    stream << "{\"x\":" << slam_info.x
           << ",\"y\":" << slam_info.y
           << ",\"heading\":" << slam_info.heading
           << ",\"v\":" << slam_info.v
           << ",\"w\":" << slam_info.w << "}";
    response->write(stream);
  };

  server.resource["^/pose_slam$"]["POST"] = [](shared_ptr<HttpServer::Response> response, shared_ptr<HttpServer::Request> request) {
    try {
      ptree pt;
      read_json(request->content, pt);
      slam_pose_lock.lock();
      slam_info.x = pt.get<float>("x");
      slam_info.y = pt.get<float>("y");
      slam_info.heading = pt.get<float>("t");
      slam_info.v = pt.get<float>("v");
      slam_info.w = pt.get<float>("w");
      slam_pose_lock.unlock();
      response->write("slam pose update recieved");
    }
    catch (const exception& e) {
      response->write(SimpleWeb::StatusCode::client_error_bad_request, e.what());
    }
  };


  server.resource["^/pose_raw$"]["GET"] = [](shared_ptr<HttpServer::Response> response, shared_ptr<HttpServer::Request> /*request*/) {
    stringstream stream;
    stream << "{\"x\":" << robot_info.x
           << ",\"y\":" << robot_info.y
           << ",\"heading\":" << robot_info.heading
           << ",\"v\":" << robot_info.v
           << ",\"w\":" << robot_info.w << "}";
    response->write(stream);
  };

  server.resource["^/raw$"]["POST"] = [](shared_ptr<HttpServer::Response> response, shared_ptr<HttpServer::Request> request) {
    try {
      ptree pt;
      read_json(request->content, pt);
      float cmd_vel = pt.get<float>("v");
      float cmd_omega = pt.get<float>("w");
      command_queue.push(RobotCommand(cmd_vel, cmd_omega));
      response->write("Command recieved");
    }
    catch (const exception& e) {
      response->write(SimpleWeb::StatusCode::client_error_bad_request, e.what());
    }
  };

  server.resource["^/stream$"]["GET"] = [](shared_ptr<HttpServer::Response> response, shared_ptr<HttpServer::Request> /*request*/) {
    thread work_thread([response] {
      SimpleWeb::CaseInsensitiveMultimap header;
      //header.emplace("transfer-encoding", "chunked");
      header.emplace("Content-Type", "multipart/x-mixed-replace; boundary=\"Webserver_JPEG_Stream\"");
      // DOOM super hackaround
      response->close_connection_after_response = true;
      response->write(SimpleWeb::StatusCode::success_ok, header);
      response->close_connection_after_response = false;
      cout << "stream::new connection" << endl;
      bool loop = true;
      std::shared_ptr<vector<uchar>> _image;

      const std::string sep = "--Webserver_JPEG_Stream";
      (*response) << sep << "\r\n";
      (*response) << "Content-Type: image/jpeg" << "\r\n\r\n";
      while (loop) {
        // TODO: bad multithreading performance lol
        image_lock.lock();
        _image = image_data;
        image_lock.unlock();
        vector<uchar>* v = _image.get();

        (*response) << std::string(v->begin(), v->end()) << "\r\n";
        (*response) << "\r\n" << sep << "\r\n";
        response->send([&loop](const error_code& c) {
            if (c.value() != 0) {
              loop = false;
            }
          });
        // cout << "send " << s << endl;
        //this_thread::sleep_for(chrono::seconds(0.1));
        (*response) << "Content-Type: image/jpeg" << "\r\n\r\n";
        usleep(100000);
      }
      cout << "stream::closing connection" << endl;
    });
    work_thread.detach();
  };

  server.resource["^/map$"]["GET"] = [](shared_ptr<HttpServer::Response> response, shared_ptr<HttpServer::Request> /*request*/) {
    thread work_thread([response] {
      SimpleWeb::CaseInsensitiveMultimap header;
      header.emplace("Content-Type", "multipart/x-mixed-replace; boundary=\"Webserver_JPEG_Stream\"");
      // DOOM super hackaround
      response->close_connection_after_response = true;
      response->write(SimpleWeb::StatusCode::success_ok, header);
      response->close_connection_after_response = false;
      cout << "map::new connection" << endl;
      bool loop = true;
      std::shared_ptr<vector<uchar>> _image;

      const std::string sep = "--Webserver_JPEG_Stream";
      (*response) << sep << "\r\n";
      (*response) << "Content-Type: image/jpeg" << "\r\n\r\n";
      while (loop) {
        // TODO: bad multithreading performance lol
        image_lock.lock();
        _image = map_data;
        image_lock.unlock();
        vector<uchar>* v = _image.get();

        (*response) << std::string(v->begin(), v->end()) << "\r\n";
        (*response) << "\r\n" << sep << "\r\n";
        response->send([&loop](const error_code& c) {
            if (c.value() != 0) {
              loop = false;
            }
          });
        // cout << "send " << s << endl;
        //this_thread::sleep_for(chrono::seconds(0.1));
        (*response) << "Content-Type: image/jpeg" << "\r\n\r\n";
        usleep(100000);
      }
      cout << "map::closing connection" << endl;
    });
    work_thread.detach();
  };

  // GET-example simulating heavy work in a separate thread
  server.resource["^/work$"]["GET"] = [](shared_ptr<HttpServer::Response> response, shared_ptr<HttpServer::Request> /*request*/) {
    thread work_thread([response] {
      this_thread::sleep_for(chrono::seconds(5));
      response->write("Work done");
    });
    work_thread.detach();
  };

  // Default GET-example. If no other matches, this anonymous function will be called.
  // Will respond with content in the web/-directory, and its subdirectories.
  // Default file: index.html
  // Can for instance be used to retrieve an HTML 5 client that uses REST-resources on this server
  server.default_resource["GET"] = [](shared_ptr<HttpServer::Response> response, shared_ptr<HttpServer::Request> request) {
    try {
      auto web_root_path = boost::filesystem::canonical("web");
      auto path = boost::filesystem::canonical(web_root_path / request->path);
      // Check if path is within web_root_path
      if(distance(web_root_path.begin(), web_root_path.end()) > distance(path.begin(), path.end()) ||
         !equal(web_root_path.begin(), web_root_path.end(), path.begin()))
        throw invalid_argument("path must be within root path");
      if(boost::filesystem::is_directory(path))
        path /= "index.html";

      SimpleWeb::CaseInsensitiveMultimap header;

      // Uncomment the following line to enable Cache-Control
      // header.emplace("Cache-Control", "max-age=86400");

#ifdef HAVE_OPENSSL
//    Uncomment the following lines to enable ETag
//    {
//      ifstream ifs(path.string(), ifstream::in | ios::binary);
//      if(ifs) {
//        auto hash = SimpleWeb::Crypto::to_hex_string(SimpleWeb::Crypto::md5(ifs));
//        header.emplace("ETag", "\"" + hash + "\"");
//        auto it = request->header.find("If-None-Match");
//        if(it != request->header.end()) {
//          if(!it->second.empty() && it->second.compare(1, hash.size(), hash) == 0) {
//            response->write(SimpleWeb::StatusCode::redirection_not_modified, header);
//            return;
//          }
//        }
//      }
//      else
//        throw invalid_argument("could not read file");
//    }
#endif

      auto ifs = make_shared<ifstream>();
      ifs->open(path.string(), ifstream::in | ios::binary | ios::ate);

      if(*ifs) {
        auto length = ifs->tellg();
        ifs->seekg(0, ios::beg);

        header.emplace("Content-Length", to_string(length));
        response->write(header);

        // Trick to define a recursive function within this scope (for example purposes)
        class FileServer {
        public:
          static void read_and_send(const shared_ptr<HttpServer::Response> &response, const shared_ptr<ifstream> &ifs) {
            // Read and send 128 KB at a time
            static vector<char> buffer(131072); // Safe when server is running on one thread
            streamsize read_length;
            if((read_length = ifs->read(&buffer[0], static_cast<streamsize>(buffer.size())).gcount()) > 0) {
              response->write(&buffer[0], read_length);
              if(read_length == static_cast<streamsize>(buffer.size())) {
                response->send([response, ifs](const SimpleWeb::error_code &ec) {
                  if(!ec)
                    read_and_send(response, ifs);
                  else
                    cerr << "Connection interrupted" << endl;
                });
              }
            }
          }
        };
        FileServer::read_and_send(response, ifs);
      }
      else
        throw invalid_argument("could not read file");
    }
    catch(const exception &e) {
      response->write(SimpleWeb::StatusCode::client_error_bad_request, "Could not open path " + request->path + ": " + e.what());
    }
  };

  server.on_error = [](shared_ptr<HttpServer::Request> /*request*/, const SimpleWeb::error_code & /*ec*/) {
    // Handle errors here
    // Note that connection timeouts will also call this handle with ec set to SimpleWeb::errc::operation_canceled
  };

  // Start server and receive assigned port when server is listening for requests
  promise<unsigned short> server_port;
  thread server_thread([&server, &server_port]() {
    // Start server
    server.start([&server_port](unsigned short port) {
      server_port.set_value(port);
    });
  });
  cout << "Server listening on port " << server_port.get_future().get() << endl
       << endl;
  
  thread robot_thread([&robot_info]() {
    // Listen to robot info.
    sd_setup("/dev/ttyAMA1");
    sd_set_speed(115200);
    sd_set_blocking();
    char recv_buf[25];
    char send_buf[9];
    send_buf[8] = '\n';
    u_int32_t index = 0;
    
    while(1) {
    
      RobotCommand new_command;
      if (command_queue.pop(new_command)) {
        //cout << "got cmd " << new_command.cmd_v << ", " << new_command.cmd_w << endl;
        *((float*)send_buf) = new_command.cmd_v * METER_TO_FEET;
        *((float*)(send_buf + 4)) = new_command.cmd_w;
        sd_writen(send_buf, 9);
      }
      sd_readn(recv_buf, 25);
      if (recv_buf[24] != '\n') {
        printf("desync %u\n", index);
        char c = 0;
        while (c != '\n') {
            int n = sd_readn(&c, 1);
        }
        sd_readn(recv_buf, 25);
        index = *((u_int32_t*) (recv_buf + 20));
        printf("sync %u\n", index);
        continue;
      }
      index += 1;
      u_int32_t new_index = *((u_int32_t*) (recv_buf + 20));
      if (new_index != index) {
          printf("bad index: %u, expected %u\n", new_index, index);
          index = new_index;
      }
      else {
        float new_x = *((float*) (recv_buf)) * FEET_TO_METER;
        float new_y = *((float*) (recv_buf + 4)) * FEET_TO_METER;
        float delta_x = new_x - robot_info.x;
        float delta_y = new_y - robot_info.y;
        float new_heading = *((float*) (recv_buf + 8));
        float delta_heading = new_heading - robot_info.heading;
        float new_v = *((float*) (recv_buf + 12)) * FEET_TO_METER;
        float new_w = *((float*) (recv_buf + 16));

        slam_pose_lock.lock();
        float heading_err = new_heading - slam_info.heading;
                float _cos = cos(heading_err);
        float _sin = sin(heading_err);
        slam_info.x += delta_x * _cos + delta_y * _sin;
        slam_info.y += delta_y * _cos - delta_x * _sin;
        slam_info.heading += delta_heading;
        slam_info.v = new_v;
        slam_info.w = new_w;
        slam_pose_lock.unlock();

        robot_info.x = new_x;
        robot_info.y = new_y;
        robot_info.heading = new_heading;
        robot_info.v = new_v;
        robot_info.w = new_w;
      }
    }
  });

  const int image_buffer_size = 10;
  cv::Mat m;
  // Image is 480x640x3 bytes.
  char* buffer = map_file(".webserver.video", O_CREAT | O_RDWR, 480*640*3);
  char* slam_buffer = map_file(".slam.map", O_CREAT | O_RDWR, 400*400*3);
  cv::Mat map_img(400, 400, CV_8UC3);
  if (buffer == NULL) {
    perror("mmap failure");
  }
  else {
    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 90};
    size_t frame = 0;
    for (;; ++frame) {
      bool res = camera.read(m);
      size_t nbytes = (m.dataend - m.datastart) * sizeof(uchar);
      memcpy(buffer, m.data, nbytes);
      
      if (frame % 3 == 0) {
        image_lock.lock();
        image_data = std::make_shared<vector<uchar>>();
        cv::imencode(".jpg", m, *image_data, params);
        image_lock.unlock();
      }
      if (frame % 30 == 0) {
        if (slam_buffer != NULL) {
          memcpy(map_img.data, slam_buffer, 400*400*3);
          map_lock.lock();
          map_data = std::make_shared<vector<uchar>>();
          cv::imencode(".jpg", map_img, *map_data, params);
          map_lock.unlock();
        }
      }
    }
  }

  server_thread.join();
}
