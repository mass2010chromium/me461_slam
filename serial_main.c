#include <stdio.h>

#include "serial_dev.c"

int main() {
    sd_setup("/dev/ttyAMA1");
    sd_set_speed(115200);
    sd_set_blocking();
    char recv_buf[21];
    u_int32_t index = 0;
    while(1) {
        sd_readn(recv_buf, 21);
        if (recv_buf[20] != '\n') {
            printf("desync %u\n", index);
            volatile char c = 0;
            while (c != '\n') {
                int n = sd_readn(&c, 1);
            }
            index = *((u_int32_t*) (recv_buf + 16));
            printf("sync %u\n", index);
            continue;
        }
        index += 1;
        u_int32_t new_index = *((u_int32_t*) (recv_buf + 16));
        if (new_index != index) {
            printf("bad index: %u, expected %u\n", new_index, index);
            index = new_index;
        }
    }
}
