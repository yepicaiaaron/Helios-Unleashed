<div align="center">

# Helios Unleashed
**Real-Time High-Quality Video Generation**

</div>

## A Special Thanks to the Original Authors

First and foremost, a massive thank you to the incredible researchers at the **[PKU-YuanGroup (Peking University)](https://github.com/PKU-YuanGroup)**. 

This project builds upon their phenomenal, groundbreaking work on the [Helios model](https://github.com/PKU-YuanGroup/Helios) and the corresponding [Helios Research Paper](https://arxiv.org/abs/2312.13400). Their open-source release of the 14-Billion parameter diffusion transformer is what makes our real-time acceleration efforts possible. We are simply pushing the limits of their phenomenal architecture for real-time WebRTC streaming and production deployments. 

## The Vision: Helios Unleashed

When the original paper launched, it showcased breathtaking, high-fidelity long video generation at >19.5 FPS on a single H100 GPU. It sounded like the holy grail of real-time multi-modal generation.

However, moving from offline batch generation to interactive, real-time deployment introduces massive engineering bottlenecks. When we deployed and tested the streaming pipeline, the 19.5 FPS claim hits reality: doing continuous, temporally consistent generation requires Image-to-Video (I2V) conditioning and VAE decoding, creating a stuttering, non-continuous playback experience.

**Helios Unleashed** is our mission to conquer these bottlenecks. We are engineering a pipeline that achieves true, continuous 24 FPS real-time generation with uncompromised cinematic quality.

### Authentic Hardware Benchmarks

We bypassed WebRTC/browser limitations to capture the authentic hardware throughput. We built a custom Python script that intercepts the raw PyTorch frames and dynamically encodes them directly to an MP4 on the server using `ffmpeg`. It programmatically duplicates frames to physically bake the exact generation delay into the video file and overlays the native hardware FPS via OpenCV.

**4-Way Side-by-Side Comparison**
*Top: Cyberpunk Detective | Bottom: Drone Metropolis*
*Left: [1, 1, 1] Steps (High Speed) | Right: [2, 2, 2] Steps (High Quality)*
<video src="assets/benchmark_hardware_quad_comparison.mp4" width="800" controls></video>

---

**Prompt: The Cyberpunk Detective**
"A cinematic low-angle tracking shot of a cyberpunk detective walking down a neon-lit alleyway in the rain, 8k, photorealistic."

*   **[1, 1, 1] Steps (High Speed):** Averages ~17-18 FPS. Smoother, but heavily degraded visual structure.
*   **[2, 2, 2] Steps (High Quality):** Averages ~11-13 FPS. Sharp details, but severe stuttering playback.

<div style="display: flex; gap: 10px; justify-content: center;">
  <div>
    <h4>[1, 1, 1] High Speed (Degraded)</h4>
    <video src="assets/benchmark_hardware_1_1_1_A.mp4" width="400" controls></video>
  </div>
  <div>
    <h4>[2, 2, 2] High Quality (Stuttering)</h4>
    <video src="assets/benchmark_hardware_2_2_2_A.mp4" width="400" controls></video>
  </div>
</div>

### Why Real-World Streaming FPS Differs from Paper Claims

Based on our analysis of the Helios-Distilled architecture, here are the major reasons why the reported 1.58s per chunk (~20.8 FPS) is faster than our PyTorch hardware readout (~2.39s per chunk / ~13.8 FPS) on the exact same H100 hardware:

1. **They benchmarked pure T2V (Text-to-Video), not I2V streaming:** Their benchmark measures just the raw transformer. Streaming requires I2V conditioning for chunk-to-chunk consistency, adding massive latency.
2. **They didn't include VAE decoding overhead:** In our streaming pipeline, we decode latents back into full 720p RGB pixels on the GPU so that we can actually stream them. The VAE decode step for 33 frames of 720p video is mathematically heavy.
3. **Stage 2 (Upscaling/Refinement):** We have `is_enable_stage2=True`. They likely disabled the second stage of the generation pipeline for their raw speed-test benchmark.
4. **Python I/O and Conversion Overhead:** Converting the `bfloat16` PyTorch tensor to `float32`, pulling it from GPU VRAM to CPU RAM (`.cpu().numpy()`), and reshaping/scaling the array takes non-trivial millisecond time.

## The First Breakthrough: RIFE Interpolation

To solve the latency vs. throughput bottleneck without sacrificing the high visual quality of the `[2, 2, 2]` setting, we introduced **RIFE (Real-Time Intermediate Flow Estimation)**.

We took the pure `[2, 2, 2]` hardware encode that was natively struggling at ~11-13 FPS, stripped out the duplicate delay frames, and ran it directly through the RIFE AI interpolation model on an L4 GPU. RIFE analyzes the 12 FPS footage, interpolates the missing frames via optical flow, and outputs a buttery-smooth **24 FPS** video with minimal delay (~10-15ms per frame).

### Before and After RIFE

Below is the visual difference. You get the stunning, photorealistic structural fidelity of the 2-step PyTorch render, combined with the real-time fluidity of a native 24 FPS stream.

**Before RIFE (~11-13 FPS natively with stutter):**
<video src="assets/benchmark_hardware_2_2_2_A.mp4" width="600" controls></video>

**After RIFE (Boosted to a buttery smooth 24 FPS):**
<video src="assets/benchmark_rife_2_2_2_A.mp4" width="600" controls></video>

## What We're Doing Next: The Future Roadmap

To transcend the current performance and achieve a true 24 FPS pipeline natively with uncompromised visual fidelity, our roadmap outlines the following engineering milestones:

### 1. B200 Hardware Acceleration
The upcoming NVIDIA Blackwell B200 GPU is central to our idea. The B200 provides **8 TB/s of HBM3e bandwidth**, solving the memory-bound limits of DiTs. Combined with FlashAttention-4 (CuTe-DSL) and TMA (Tensor Memory Accelerator), the B200 will shatter current compilation walls and tensor throughput limits.

### 2. Multi-GPU Asynchronous Pipelining & Zero-Copy NVENC
We are building a **3-Stage Async Pipeline** across GPUs using Pipeline Parallelism:
*   **GPU 0 (The Brain):** Exclusively runs the Helios Transformer generating latents.
*   **GPU 1 (The Painter & Smoother):** VAE Decoder and RIFE Interpolation running asynchronously via NVLink.
*   **Zero-Copy NVENC:** Passing PyTorch tensor pointers directly to NVIDIA's hardware encoder (NVENC) to completely eliminate Python `.cpu().numpy()` I/O overhead.

### 3. LightX2V Integration for Extreme Performance (FP8/NVFP4 Quantisation)
[LightX2V](https://github.com/ModelTC/LightX2V) is an advanced inference framework designed for visual generation.
*   **FP8/NVFP4 Support:** We will leverage LightX2V's implementations for FP8/NVFP4 inference (like `Self-Forcing-FP8`). Quantizing the model to FP8 doubles matrix multiplication throughput and halves the VRAM footprint. This allows the `[2, 2, 2]` setting to run as fast as the low-quality `[1, 1, 1]` setting.

### 4. GenRL for Advanced Motion and Aesthetic Tuning
When dropping Helios to the `[1, 1, 1]` speed-test setting, the quality becomes smeared and glitchy. [GenRL](https://github.com/ModelTC/GenRL) is the missing piece for solving the visual quality problem.
*   **FlowGRPO Reinforcement Learning:** We will run the Helios-Distilled model through GenRL using built-in `videoalign_mq` (Motion Quality) and `hpsv3_general` (Aesthetics) reward functions. We can effectively fine-tune the model to look structurally coherent and drastically better even at the `[1, 1, 1]` step count.

By combining the B200's raw throughput, LightX2V's FP8 engine, GenRL's reward tuning, and RIFE's optical flow interpolation, we will hit the holy grail: **24 FPS real-time generation that actually looks like the paper's original high-quality benchmarks.**

---

**Join Us on this Journey.** If you're an engineer obsessed with killing latency, writing Triton kernels, and building the future of real-time AI generation, we're building the ultimate foundation right here.
