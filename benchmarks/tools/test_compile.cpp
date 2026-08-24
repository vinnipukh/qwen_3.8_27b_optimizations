#include "llama.h"
#include "common.h"
#include "arg.h"
#include "log.h"
#include <iostream>
#include <chrono>

int main() {
    llama_backend_init();
    std::cout << "Compiled and linked successfully!" << std::endl;
    llama_backend_free();
    return 0;
}
