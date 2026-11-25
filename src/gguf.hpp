// Minimal GGUF v3 reader (mmap). Only what this engine needs: typed metadata, string arrays, tensor table.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace gguf {

enum { T_F32 = 0, T_F16 = 1, T_Q8_0 = 8, T_BF16 = 30 };
// Fork-private type id of PTQ1_0 (28 bytes per 128 weights) is discovered at load (see engine).

struct Tensor { std::string name; std::vector<int64_t> ne; uint32_t type = 0; uint64_t off = 0; const uint8_t* data = nullptr; };

struct Val { uint32_t type = 0; double num = 0; std::string str; std::vector<std::string> strs; std::vector<double> nums; };

struct File {
    std::unordered_map<std::string, Val> kv;
    std::vector<Tensor> tensors;
    std::unordered_map<std::string, size_t> index;
    const uint8_t* base = nullptr; size_t size = 0; int fd = -1;

    void open(const char* path) {
        fd = ::open(path, O_RDONLY); if (fd < 0) throw std::runtime_error("cannot open gguf");
        struct stat st; fstat(fd, &st); size = st.st_size;
        base = (const uint8_t*) mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (base == MAP_FAILED) throw std::runtime_error("mmap failed");
        madvise((void*) base, size, MADV_SEQUENTIAL);
        const uint8_t* p = base; const uint8_t* end = base + size;
        auto rd = [&](auto& v) { if (p + sizeof(v) > end) throw std::runtime_error("truncated gguf"); memcpy(&v, p, sizeof(v)); p += sizeof(v); };
        auto rstr = [&]() { uint64_t n; rd(n); if (n > (uint64_t)(end - p)) throw std::runtime_error("bad string"); std::string s((const char*) p, n); p += n; return s; };
        uint32_t magic, ver; rd(magic); rd(ver); if (magic != 0x46554747 || ver != 3) throw std::runtime_error("not GGUF v3");
        uint64_t nt, nk; rd(nt); rd(nk);
        auto rnum = [&](uint32_t t) -> double {
            switch (t) {
                case 0: { uint8_t v; rd(v); return v; } case 1: { int8_t v; rd(v); return v; }
                case 2: { uint16_t v; rd(v); return v; } case 3: { int16_t v; rd(v); return v; }
                case 4: { uint32_t v; rd(v); return v; } case 5: { int32_t v; rd(v); return v; }
                case 6: { float v; rd(v); return v; } case 7: { uint8_t v; rd(v); return v; }
                case 10: { uint64_t v; rd(v); return (double) v; } case 11: { int64_t v; rd(v); return (double) v; }
                case 12: { double v; rd(v); return v; }
            }
            throw std::runtime_error("bad kv type");
        };
        for (uint64_t i = 0; i < nk; ++i) {
            std::string k = rstr(); uint32_t t; rd(t); Val v; v.type = t;
            if (t == 8) v.str = rstr();
            else if (t == 9) {
                uint32_t et; uint64_t n; rd(et); rd(n); v.type = et | 0x100;
                for (uint64_t j = 0; j < n; ++j) { if (et == 8) v.strs.push_back(rstr()); else v.nums.push_back(rnum(et)); }
            } else v.num = rnum(t);
            kv.emplace(std::move(k), std::move(v));
        }
        for (uint64_t i = 0; i < nt; ++i) {
            Tensor t; t.name = rstr(); uint32_t nd; rd(nd);
            for (uint32_t d = 0; d < nd; ++d) { uint64_t v; rd(v); t.ne.push_back((int64_t) v); }
            rd(t.type); rd(t.off); index[t.name] = tensors.size(); tensors.push_back(t);
        }
        uint64_t al = kv.count("general.alignment") ? (uint64_t) kv["general.alignment"].num : 32;
        uint64_t data = ((uint64_t)(p - base) + al - 1) / al * al;
        for (auto& t : tensors) t.data = base + data + t.off;
    }
    const Tensor* find(const std::string& n) const { auto it = index.find(n); return it == index.end() ? nullptr : &tensors[it->second]; }
    const Tensor& get(const std::string& n) const { auto* t = find(n); if (!t) throw std::runtime_error("missing tensor " + n); return *t; }
    double num(const std::string& k, double def = 0) const { auto it = kv.find(k); return it == kv.end() ? def : it->second.num; }
    ~File() { if (base) munmap((void*) base, size); if (fd >= 0) ::close(fd); }
};

} // namespace gguf
