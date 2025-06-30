#define ENABLE_VHACD_IMPLEMENTATION 1
#include "VHACD.h"
#include "vhacd_wrapper.h"
#include <memory>
#include <vector>

struct VHACDWrapper {
    VHACD::IVHACD* vhacd;
    VHACD::IVHACD::Parameters params;
    std::vector<VHACD::IVHACD::ConvexHull> cached_hulls;  // Cache hulls to keep data alive
    
    VHACDWrapper(VHACD::IVHACD* instance) : vhacd(instance) {}
    ~VHACDWrapper() {
        if (vhacd) {
            vhacd->Release();
        }
    }
};

extern "C" {

VHACDHandle vhacd_create(void) {
    VHACD::IVHACD* vhacd = VHACD::CreateVHACD();
    if (!vhacd) return nullptr;
    return new VHACDWrapper(vhacd);
}

VHACDHandle vhacd_create_async(void) {
    VHACD::IVHACD* vhacd = VHACD::CreateVHACD_ASYNC();
    if (!vhacd) return nullptr;
    return new VHACDWrapper(vhacd);
}

void vhacd_release(VHACDHandle handle) {
    if (handle) {
        delete static_cast<VHACDWrapper*>(handle);
    }
}

void vhacd_set_max_convex_hulls(VHACDHandle handle, unsigned int max_convex_hulls) {
    if (handle) {
        static_cast<VHACDWrapper*>(handle)->params.m_maxConvexHulls = max_convex_hulls;
    }
}

void vhacd_set_resolution(VHACDHandle handle, unsigned int resolution) {
    if (handle) {
        static_cast<VHACDWrapper*>(handle)->params.m_resolution = resolution;
    }
}

void vhacd_set_minimum_volume_percent_error_allowed(VHACDHandle handle, double min_volume_percent) {
    if (handle) {
        static_cast<VHACDWrapper*>(handle)->params.m_minimumVolumePercentErrorAllowed = min_volume_percent;
    }
}

void vhacd_set_max_recursion_depth(VHACDHandle handle, unsigned int max_depth) {
    if (handle) {
        static_cast<VHACDWrapper*>(handle)->params.m_maxRecursionDepth = max_depth;
    }
}

void vhacd_set_shrink_wrap(VHACDHandle handle, int enable_shrink_wrap) {
    if (handle) {
        static_cast<VHACDWrapper*>(handle)->params.m_shrinkWrap = enable_shrink_wrap != 0;
    }
}

void vhacd_set_fill_mode(VHACDHandle handle, VHACDFillMode fill_mode) {
    if (handle) {
        VHACD::FillMode mode;
        switch (fill_mode) {
            case VHACD_FILL_FLOOD_FILL:
                mode = VHACD::FillMode::FLOOD_FILL;
                break;
            case VHACD_FILL_SURFACE_ONLY:
                mode = VHACD::FillMode::SURFACE_ONLY;
                break;
            case VHACD_FILL_RAYCAST_FILL:
                mode = VHACD::FillMode::RAYCAST_FILL;
                break;
            default:
                mode = VHACD::FillMode::FLOOD_FILL;
                break;
        }
        static_cast<VHACDWrapper*>(handle)->params.m_fillMode = mode;
    }
}

void vhacd_set_max_num_vertices_per_ch(VHACDHandle handle, unsigned int max_vertices) {
    if (handle) {
        static_cast<VHACDWrapper*>(handle)->params.m_maxNumVerticesPerCH = max_vertices;
    }
}

void vhacd_set_async_acd(VHACDHandle handle, int enable_async) {
    if (handle) {
        static_cast<VHACDWrapper*>(handle)->params.m_asyncACD = enable_async != 0;
    }
}

void vhacd_set_min_edge_length(VHACDHandle handle, unsigned int min_edge_length) {
    if (handle) {
        static_cast<VHACDWrapper*>(handle)->params.m_minEdgeLength = min_edge_length;
    }
}

void vhacd_set_find_best_plane(VHACDHandle handle, int enable_best_plane) {
    if (handle) {
        static_cast<VHACDWrapper*>(handle)->params.m_findBestPlane = enable_best_plane != 0;
    }
}

int vhacd_compute_float(VHACDHandle handle, 
                        const float* points, 
                        unsigned int count_points,
                        const unsigned int* triangles, 
                        unsigned int count_triangles) {
    if (!handle) return 0;
    VHACDWrapper* wrapper = static_cast<VHACDWrapper*>(handle);
    return wrapper->vhacd->Compute(points, count_points, triangles, count_triangles, wrapper->params) ? 1 : 0;
}

int vhacd_compute_double(VHACDHandle handle, 
                         const double* points, 
                         unsigned int count_points,
                         const unsigned int* triangles, 
                         unsigned int count_triangles) {
    if (!handle) return 0;
    VHACDWrapper* wrapper = static_cast<VHACDWrapper*>(handle);
    return wrapper->vhacd->Compute(points, count_points, triangles, count_triangles, wrapper->params) ? 1 : 0;
}

unsigned int vhacd_get_n_convex_hulls(VHACDHandle handle) {
    if (!handle) return 0;
    return static_cast<VHACDWrapper*>(handle)->vhacd->GetNConvexHulls();
}

int vhacd_get_convex_hull(VHACDHandle handle, 
                          unsigned int index,
                          double** out_points,
                          unsigned int* out_n_points,
                          unsigned int** out_triangles,
                          unsigned int* out_n_triangles) {
    if (!handle) return 0;
    
    VHACDWrapper* wrapper = static_cast<VHACDWrapper*>(handle);
    
    // Ensure we have cached the hulls
    if (wrapper->cached_hulls.empty()) {
        unsigned int n_hulls = wrapper->vhacd->GetNConvexHulls();
        wrapper->cached_hulls.resize(n_hulls);
        
        for (unsigned int i = 0; i < n_hulls; ++i) {
            if (!wrapper->vhacd->GetConvexHull(i, wrapper->cached_hulls[i])) {
                wrapper->cached_hulls.clear();
                return 0;
            }
        }
    }
    
    if (index >= wrapper->cached_hulls.size()) {
        return 0;
    }
    
    const auto& ch = wrapper->cached_hulls[index];
    
    // Return pointers to the cached data (now stays valid until wrapper is destroyed)
    *out_points = const_cast<double*>(reinterpret_cast<const double*>(ch.m_points.data()));
    *out_n_points = static_cast<unsigned int>(ch.m_points.size());
    *out_triangles = const_cast<unsigned int*>(reinterpret_cast<const unsigned int*>(ch.m_triangles.data()));
    *out_n_triangles = static_cast<unsigned int>(ch.m_triangles.size());
    
    return 1;
}

int vhacd_get_convex_hull_volume(VHACDHandle handle, unsigned int index, double* volume) {
    if (!handle) return 0;
    
    VHACDWrapper* wrapper = static_cast<VHACDWrapper*>(handle);
    VHACD::IVHACD::ConvexHull ch;
    
    if (!wrapper->vhacd->GetConvexHull(index, ch)) {
        return 0;
    }
    
    *volume = ch.m_volume;
    return 1;
}

int vhacd_get_convex_hull_center(VHACDHandle handle, unsigned int index, double center[3]) {
    if (!handle) return 0;
    
    VHACDWrapper* wrapper = static_cast<VHACDWrapper*>(handle);
    VHACD::IVHACD::ConvexHull ch;
    
    if (!wrapper->vhacd->GetConvexHull(index, ch)) {
        return 0;
    }
    
    center[0] = ch.m_center.GetX();
    center[1] = ch.m_center.GetY();
    center[2] = ch.m_center.GetZ();
    return 1;
}

int vhacd_get_convex_hull_bounds(VHACDHandle handle, unsigned int index, 
                                 double min_bounds[3], double max_bounds[3]) {
    if (!handle) return 0;
    
    VHACDWrapper* wrapper = static_cast<VHACDWrapper*>(handle);
    VHACD::IVHACD::ConvexHull ch;
    
    if (!wrapper->vhacd->GetConvexHull(index, ch)) {
        return 0;
    }
    
    min_bounds[0] = ch.mBmin.GetX();
    min_bounds[1] = ch.mBmin.GetY();
    min_bounds[2] = ch.mBmin.GetZ();
    max_bounds[0] = ch.mBmax.GetX();
    max_bounds[1] = ch.mBmax.GetY();
    max_bounds[2] = ch.mBmax.GetZ();
    return 1;
}

int vhacd_is_ready(VHACDHandle handle) {
    if (!handle) return 0;
    return static_cast<VHACDWrapper*>(handle)->vhacd->IsReady() ? 1 : 0;
}

void vhacd_cancel(VHACDHandle handle) {
    if (handle) {
        static_cast<VHACDWrapper*>(handle)->vhacd->Cancel();
    }
}

int vhacd_compute_center_of_mass(VHACDHandle handle, double center_of_mass[3]) {
    if (!handle) return 0;
    return static_cast<VHACDWrapper*>(handle)->vhacd->ComputeCenterOfMass(center_of_mass) ? 1 : 0;
}

unsigned int vhacd_find_nearest_convex_hull(VHACDHandle handle, 
                                           const double pos[3], 
                                           double* distance_to_hull) {
    if (!handle) return 0;
    return static_cast<VHACDWrapper*>(handle)->vhacd->findNearestConvexHull(pos, *distance_to_hull);
}

void vhacd_clean(VHACDHandle handle) {
    if (handle) {
        VHACDWrapper* wrapper = static_cast<VHACDWrapper*>(handle);
        wrapper->vhacd->Clean();
        wrapper->cached_hulls.clear();  // Clear cached hulls when cleaning
    }
}

} // extern "C"