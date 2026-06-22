// Metal context: the single MTLDevice / default command queue, the compute-pipeline cache, the
// pointer->MTLBuffer registry (so the runtime's void* device pointers map back to bindable buffers),
// and the default.metallib loader (found next to this dylib via dladdr). Implements the mtl:: seam.
#import "internal.h"
#import <Foundation/Foundation.h>
#include <dlfcn.h>
#include <map>
#include <unordered_map>
#include <mutex>
#include <string>

namespace {
struct BufRec { id<MTLBuffer> buf; size_t len; };
struct Ctx {
    id<MTLDevice> dev = nil;
    id<MTLLibrary> lib = nil;
    id<MTLCommandQueue> queue = nil;
    std::mutex mu;
    std::map<uintptr_t, BufRec> bufs;                                  // base addr -> {buffer, length}
    std::unordered_map<std::string, id<MTLComputePipelineState>> pso;  // function name -> pipeline
};
Ctx &C() { static Ctx c; return c; }

// Locate default.metallib next to the loaded libascendcl.dylib (robust regardless of CWD / how it's linked).
NSString *metallib_path() {
    Dl_info info;
    if (dladdr((const void *)&metallib_path, &info) && info.dli_fname) {
        NSString *self = [NSString stringWithUTF8String:info.dli_fname];
        return [[self stringByDeletingLastPathComponent] stringByAppendingPathComponent:@"default.metallib"];
    }
    return @"default.metallib";
}
} // namespace

namespace mtl {

void init() {
    std::lock_guard<std::mutex> lk(C().mu);
    if (C().dev) return;
    C().dev = MTLCreateSystemDefaultDevice();
    if (!C().dev) { fprintf(stderr, "[metal] no Metal device available\n"); return; }
    C().queue = [C().dev newCommandQueue];
    NSError *err = nil;
    NSString *p = metallib_path();
    C().lib = [C().dev newLibraryWithURL:[NSURL fileURLWithPath:p] error:&err];
    if (!C().lib) {
        fprintf(stderr, "[metal] failed to load %s: %s\n",
                p.UTF8String, err ? err.localizedDescription.UTF8String : "unknown");
    }
}

id<MTLDevice> device() { return C().dev; }
id<MTLCommandQueue> defaultQueue() { return C().queue; }

id<MTLComputePipelineState> pipeline(NSString *name) {
    std::lock_guard<std::mutex> lk(C().mu);
    std::string key = name.UTF8String;
    auto it = C().pso.find(key);
    if (it != C().pso.end()) return it->second;
    if (!C().lib) return nil;
    id<MTLFunction> fn = [C().lib newFunctionWithName:name];
    if (!fn) { fprintf(stderr, "[metal] kernel not found: %s\n", key.c_str()); return nil; }
    NSError *err = nil;
    id<MTLComputePipelineState> p = [C().dev newComputePipelineStateWithFunction:fn error:&err];
    if (!p) { fprintf(stderr, "[metal] pipeline build failed for %s: %s\n",
                      key.c_str(), err ? err.localizedDescription.UTF8String : "unknown"); return nil; }
    C().pso[key] = p;
    return p;
}

void *alloc(size_t n) {
    if (n == 0) n = 1;
    id<MTLBuffer> b = [C().dev newBufferWithLength:n options:MTLResourceStorageModeShared];
    if (!b) return nullptr;
    void *p = b.contents;
    std::lock_guard<std::mutex> lk(C().mu);
    C().bufs[(uintptr_t)p] = BufRec{b, n};   // ARC retains the buffer while it lives in the map
    return p;
}

void free_(void *p) {
    if (!p) return;
    std::lock_guard<std::mutex> lk(C().mu);
    C().bufs.erase((uintptr_t)p);            // ARC releases the buffer
}

id<MTLBuffer> bufferFor(const void *p, size_t *byteOffset) {
    std::lock_guard<std::mutex> lk(C().mu);
    uintptr_t addr = (uintptr_t)p;
    auto it = C().bufs.upper_bound(addr);    // first base strictly greater than addr
    if (it == C().bufs.begin()) return nil;
    --it;                                    // greatest base <= addr
    uintptr_t base = it->first;
    if (addr < base || addr >= base + it->second.len) return nil;
    if (byteOffset) *byteOffset = addr - base;
    return it->second.buf;
}

} // namespace mtl
