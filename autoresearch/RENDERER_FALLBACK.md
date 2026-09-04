# Renderer fallback: getting pixels without GL_ARB_bindless_texture

Status: investigation complete, recommendation below. Written 2026-09-03
during the first overnight run. No code changed on the Studio side - this
is a decision document.

## The wall

`standard_fragment.glsl` and `pbr_fragment.glsl` both start with
`#extension GL_ARB_bindless_texture : require`. Every material carries
bindless texture handles (`uvec2`) in an SSBO
(`MaterialGPU.texture_handles`, src/core/ecs/graphics/OpenGL.zig), and the
fragment shader constructs samplers from those handles inline:

    sampler2D diffuseTex = sampler2D(mat.texture_handles[TEX_BASE_COLOR]);

No CPU software rasterizer implements GL_ARB_bindless_texture - not
llvmpipe, not softpipe, not SWR (checked against Mesa's feature matrix;
llvmpipe caps at GL 4.5 and bindless is not on its roadmap). On the
Railway CPU box this is a hard stop: shader compile fails, no pixels.
This is why milestone 1 is physics-only.

## Why the renderer is built this way

Bindless lets one instanced draw call span arbitrarily many textures: the
material SSBO carries handles, so the shader never rebinds. It is the
right design for an interactive editor with big mixed-material scenes.
Any fallback trades that away; the question is how much the training
workload actually needs.

## Options

### A. Texture arrays (the proper fix)

Pack textures into `sampler2DArray` atlases-by-size at load time (bucket
by dimensions, e.g. one array per power-of-two size). MaterialGPU carries
layer indices (u32) instead of u64 handles; the shader change is small:

    texture(baseColorArray, vec3(TexCoord, float(mat.texture_indices[TEX_BASE_COLOR])))

- Shader side: ~6 lines per shader, GLSL 4.5-safe (array layer is a
  texture coordinate, not a dynamic sampler index - fully legal).
- Zig side: ResourceManager.zig gains array-pool allocation instead of
  makeBindless(); OpenGL.zig MaterialGPU layout changes (u32 indices,
  std430-friendly); upload path writes layers with glTexSubImage3D.
- Cost: a focused patch to 3 files plus the 2 shaders. Risk: textures of
  many distinct sizes need bucketing; repeat/wrap modes still fine per
  array; mipmaps fine.
- Keeps single-draw-call batching. Works on llvmpipe 4.5.

### B. Texture atlas (rejected)

One big atlas, materials carry UV rects. Breaks tiled/repeated UVs
(wrapping leaks across atlas cells) and bleeds across mip levels. The
procedural environments tile textures. Not viable.

### C. Per-draw texture binding (rejected)

Bind N conventional samplers, split draw calls by texture set, index
samplers only with dynamically-uniform expressions. Legal in 4.5 but
destroys the batching the renderer is built around, and "dynamically
uniform" indexing is a portability minefield across drivers. More
surgery than A for a worse result.

### D. Purpose-built software rasterizer for training (recommended for CPU)

Vision policies for navigation do not need photorealism - depth +
semantic segmentation is the standard training signal (and what most
sim-to-real quad work uses). The auto-researcher's procedural scenes are
analytic primitives (boxes/cylinders on a floor plane): a small
CPU rasterizer - ray-cast or scanline over the primitive list, ~300 lines
of Zig with zero GL - produces depth + segmentation frames at training
rates, deterministically, with no driver in the loop at all.

- No dependency on the editor renderer; cannot break the Studio app.
- Perfectly paired with the physics-only headless episode API
  (HEADLESS_API.md): same process, same step function, extra observation
  channel.
- Downside: not the real render pipeline, so a photorealism gap remains
  for final sim-to-real. That gap is exactly what option A would close
  later.

### E. GPU box (rejected for now)

Railway does not offer GPU instances (tracked feature request, not
shipped). A GPU elsewhere (RunPod/Lambda) runs the real renderer
unmodified but adds a second provider, sync story, and hourly cost, and
still doesn't fix CPU-side dev. Revisit if photoreal training becomes
the bottleneck.

## Recommendation

1. Milestone 1 stays physics-only (HEADLESS_API.md).
2. Milestone 1.5: option D - software depth/segmentation rasterizer over
   the procedural primitives, inside the headless episode API. Unblocks
   vision-based training on the CPU box with zero renderer risk.
3. When photoreal fidelity matters: option A (texture arrays) as a
   reviewed patch to Studio - it is the smallest change that removes the
   bindless requirement everywhere, including future headless machines.

Open questions for Eyad:
- Depth+segmentation sufficient for the first vision policy, or is RGB a
  hard requirement from the start?
- If A: acceptable to cap texture size buckets (e.g. everything resized
  into 256/1024/4096 arrays) in the editor's resource pipeline?
