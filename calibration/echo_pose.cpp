#include <simple-web-server/client_http.hpp>
#include <future>

#include <iostream>

#define BOOST_SPIRIT_THREADSAFE
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

using namespace boost::property_tree;

using HttpClient = SimpleWeb::Client<SimpleWeb::HTTP>;

using namespace std;

int main() {
    while(true) {
        cout << "requesting pose" << endl;
        HttpClient client("localhost:8080");
        client.request("GET", "/pose", "", [](shared_ptr<HttpClient::Response> response, const SimpleWeb::error_code& ec) {
            if (!ec) {
                cout << "Response content: " << response->content.rdbuf() << endl;
            }
            else {
                cout << "failed" << endl;
            }
        });
        client.io_service->run();
        sleep(1);
    }
}
