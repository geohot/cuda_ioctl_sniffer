// credit https://gist.githubusercontent.com/lunixbochs/03f24eab46d7eaa79a9e78aebe13a2e9/raw/1ce6ca029f2284a6189f1e734814242b73cc0cc6/shadow.cc

#include <vector>
#include <sstream>
#include <string>
#include <iostream>
#include <fstream>
#include <cstring>
#include <sys/mman.h>

namespace shadow {
    struct Map {
        uintptr_t addr;
        size_t size;
        int prot;
        std::string name;
        std::vector<uint8_t> shadow;

        Map(uintptr_t addr, size_t size, int prot, std::string name) :
            addr(addr), size(size), prot(prot), name(name) {
                this->save();
        }

        void save() {
            if (!(this->prot & PROT_READ)) return;
            if (this->name == "[vsyscall]" || this->name == "[vvar]") return;
            if (this->addr >= 0x800000000000) return;
            if (this->size > 0x10000000000) return; // 1TB
            printf("+ %s\n", this->str().c_str());
            this->shadow.resize(this->size);
            memcpy(&this->shadow[0], (void *)this->addr, this->size);
        }

        bool changed() {
            if (this->shadow.size() < this->size) return false;
            uint32_t *real = (uint32_t *)this->addr;
            bool ret = false;
            for (ssize_t i = 0; i < this->size/4; i++) {
                if (this->shadow[i] != real[i]) {
                    printf("%p: %x -> %x\n", &real[i], this->shadow[i], real[i]);
                    ret = true;
                }
            }
            return ret;
        }

        std::string str() {
            std::ostringstream os;
            os << std::hex << this->addr << " " << this->size << " ";
            os << ((this->prot & PROT_READ)  ? "r" : "-");
            os << ((this->prot & PROT_WRITE) ? "w" : "-");
            os << ((this->prot & PROT_EXEC)  ? "x" : "-");
            os << " " << this->name;
            return os.str();
        }
    };

    void diff(std::vector<Map> &maps) {
        for (auto &map : maps) {
            auto desc = map.str();
            if (map.shadow.empty()) {
                printf("! %s\n", desc.c_str());
            } else if (! map.changed()) {
                printf("  %s\n", desc.c_str());
            } else {
                printf("~ %s\n", desc.c_str());
            }
        }
    }

    std::vector<Map> get(std::string needle = "") {
        std::vector<Map> maps;
        std::ifstream f("/proc/self/maps");
        std::string line;
        while (std::getline(f, line)) {
            char *path = NULL;
            char *prot_str = NULL;
            uint64_t start = 0, end = 0;
            if (0 >= sscanf(line.c_str(), "%lx-%lx %ms %*x %*d:%*d %*d %ms\n", &start, &end, &prot_str, &path)) break;
            int prot = 0;
            if (strstr(prot_str, "r")) prot |= PROT_READ;
            if (strstr(prot_str, "w")) prot |= PROT_WRITE;
            if (strstr(prot_str, "x")) prot |= PROT_EXEC;
            if (needle.empty() || path && strstr(path, needle.c_str()) != NULL) {
                maps.emplace_back(start, end - start, prot, path ? path : "");
            }
            free(path);
        }
        return maps;
    }
}

