#include <stdio.h>
#include <stdint.h>

__constant__ char SMART_CHARSET[95] = {
    // Lowercase (0-25)
    'a','b','c','d','e','f','g','h','i','j','k','l','m',
    'n','o','p','q','r','s','t','u','v','w','x','y','z',
    // Digits (26-35)
    '0','1','2','3','4','5','6','7','8','9',
    // Uppercase (36-61)
    'A','B','C','D','E','F','G','H','I','J','K','L','M',
    'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    // Symbols (62-94)
    '_','-','!','@','#','$','%','^','&','*','(',')','+','=',
    '[',']','{','}','|',';',':','\'','"','<','>',',','.',
    '?','/','\\','`','~',' '
};

__global__ void test_password_generation() {
    printf("Testing password generation for indices 0-9:\n\n");
    
    for(uint64_t password_index = 0; password_index < 10; password_index++) {
        uint8_t temp_pass[16];
        int len = 0;
        
        if (password_index == 0) {
            temp_pass[0] = SMART_CHARSET[0]; // 'a'
            len = 1;
        } else {
            uint64_t temp = password_index;
            while (temp > 0 && len < 16) {
                temp_pass[len++] = SMART_CHARSET[temp % 95];
                temp /= 95;
            }
        }
        
        printf("Index %llu: '", password_index);
        for(int i=0; i<len; i++) {
            printf("%c", temp_pass[i]);
        }
        printf("' (len=%d)\n", len);
    }
}

int main() {
    test_password_generation<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}

