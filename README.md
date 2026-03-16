<div align="center">

# Helios B200 Unleashed: Real-Time High-Quality Video Generation

</div>

## The Promise vs. The Brutal Reality

When the original [Helios paper](https://arxiv.org/abs/2312.13400) and [repository](https://github.com/PKU-YuanGroup/Helios) were released, we were incredibly excited. The benchmarks showcased breathtaking, high-fidelity long video generation. A 14-Billion parameter diffusion transformer generating video at >19.5 FPS on a single H100 GPU? No sharding? No massive server clusters? It sounded like the holy grail of real-time multi-modal generation.

However, our excitement turned into severe disappointment over the weekend when we finally deployed and tested the interactive, real-time generation ourselves. We discovered a massive, disheartening gap between offline batch generation and interactive, real-time deployment. 

Here is exactly what we found when we went deep into the technical architecture, and why we are now completely rewriting the system for the B200.

### The "33-Frame Chunk" Codebase Problem and I2V Overhead

The deepest flaw we uncovered is in the core codebase itself. The native `pipeline_helios_diffusers.py` was built for offline batch generation. It calculates and waits for all 240 frames to finish before it outputs the video tensor.

To make it stream over WebRTC for a live user, we had to write a custom patch to mutilate their autoregressive loop. Instead of waiting for 240 frames, we forced the model to `yield` the video the split second a **33-frame chunk** passes through the VAE decoder. 

Our pipeline runs a continuous autoregressive stream where it has to pass the previous chunk's frames back into the model as an **Image-to-Video (I2V)** conditioning step to maintain consistency across chunks. That extra conditioning pass adds significant overhead. 

Because the H100 memory limits (pulling 46.5 GB VRAM for 33 frames) and the lack of a KV-cache force us to batch render these chunks and run I2V conditioning, the playback on the client side is fundamentally broken. You receive 33 frames of video, followed by a hard freeze while the GPU sweats to render the next batch through the VAE, and then you get the next 33 frames. It is a chunky, jittery, non-continuous playback experience. 

### Why Real-World FPS Differs from Paper Claims (The True Benchmarks)

Based on our analysis and testing of the Helios-Distilled architecture, there are major reasons why the reported numbers in the research paper (1.58s per chunk / ~20.8 FPS) are faster than our pure PyTorch hardware readout (~2.39s per chunk / ~13.8 FPS) on the exact same NVIDIA H100 hardware:

1. **They benchmarked pure T2V (Text-to-Video), not I2V streaming:** Their benchmark measures just the raw transformer generating latents from text. As described above, streaming requires I2V conditioning for chunk-to-chunk consistency, adding massive latency.
2. **They didn't include VAE decoding overhead:** Their 19.5+ FPS benchmark only timed the latent generation. In our streaming pipeline, we are physically decoding those latents back into full 720p RGB pixels on the GPU so that we can actually see and stream them. The VAE decode step for 33 frames of 720p video is mathematically heavy.
3. **Stage 2 (Upscaling/Refinement):** In our `stream_helios_generator_authentic.py` script, we have `is_enable_stage2=True`. They likely disabled the second stage of the generation pipeline for their raw speed-test benchmark to hit that sub-1.6s number.
4. **Python I/O and Conversion Overhead:** Our script isn't just generating the tensor; it's converting the `bfloat16` PyTorch tensor to `float32`, pulling it from GPU VRAM to CPU RAM (`.cpu().numpy()`), and reshaping/scaling the array so we can feed it into LiveKit or `ffmpeg`. Moving a 33-frame 720p uncompressed tensor from GPU to CPU takes non-trivial millisecond time.

**Essentially, they benchmarked the absolute bare-minimum matrix multiplication inside the Transformer. We are benchmarking the actual, usable pipeline that produces pixels you can look at.**

### Authentic Hardware Benchmarks

We bypassed WebRTC/browser limitations to capture the authentic hardware throughput. We built a custom Python script that intercepts the raw PyTorch frames and dynamically encodes them directly to an MP4 on the server using `ffmpeg`. It programmatically duplicates frames to physically bake the exact generation delay into the video file and overlays the native hardware FPS via OpenCV.

**Prompt A: The Cyberpunk Detective**
"A cinematic low-angle tracking shot of a cyberpunk detective walking down a neon-lit alleyway in the rain, 8k, photorealistic."

*   **[1, 1, 1] Steps (High Speed):** Averages ~17-18 FPS. Smoother, but heavily degraded visual structure.
*   **[2, 2, 2] Steps (High Quality):** Averages ~11-13 FPS. Sharp details, but severe stuttering playback.

<div style="display: flex; gap: 10px;">
  <video src="assets/benchmark_hardware_1_1_1_A.mp4" width="400" controls></video>
  <video src="assets/benchmark_hardware_2_2_2_A.mp4" width="400" controls></video>
</div>

## The Solution: RIFE Interpolation

To solve the latency vs throughput bottleneck without sacrificing the high visual quality of the `[2, 2, 2]` setting, we introduce **RIFE (Real-Time Intermediate Flow Estimation)** into an asynchronous multi-GPU pipeline.

We took the pure `[2, 2, 2]` hardware encode that was natively struggling at ~11-13 FPS, stripped out the duplicate delay frames, and ran it directly through the RIFE AI interpolation model on an L4 GPU. RIFE analyzes the 12 FPS footage, interpolates the missing frames via optical flow, and outputs a buttery-smooth **24 FPS** video with minimal delay (~10-15ms per frame).

### Before and After RIFE

Below is the visual difference. You get the stunning, photorealistic structural fidelity of the 2-step PyTorch render, combined with the real-time fluidity of a native 24 FPS stream.

**Before RIFE (~11-13 FPS natively with stutter):**
<video src="assets/benchmark_hardware_2_2_2_A.mp4" width="600" controls></video>

**After RIFE (Boosted to a buttery smooth 24 FPS):**
<video src="assets/benchmark_rife_2_2_2_A.mp4" width="600" controls></video>


## Future Roadmap: 24 FPS with High Visual Quality on B200

To transcend the current performance and achieve a true 24 FPS pipeline natively on the upcoming NVIDIA B200 GPU, our roadmap focuses on the following integration pathways:

### 1. B200 Hardware & NVENC Zero-Copy Pipeline
The B200 provides **8 TB/s of HBM3e bandwidth**, solving the memory-bound limits of DiTs. Furthermore, we will build a **3-Stage Async Pipeline** across GPUs using Pipeline Parallelism:
*   **GPU 0 (The Brain):** Helios Transformer running `[2, 2, 2]` latents natively.
*   **GPU 1 (The Painter & Smoother):** VAE Decoder and RIFE Interpolation running asynchronously via NVLink.
*   **Zero-Copy NVENC:** Passing PyTorch tensor pointers directly to NVIDIA's hardware encoder (NVENC) to completely eliminate Python `.cpu().numpy()` I/O overhead.

### 2. LightX2V Integration for Extreme Performance (FP8/NVFP4 Quantisation)
LightX2V is an advanced inference framework designed for visual generation.
*   **FP8/NVFP4 Support:** We will leverage LightX2V's implementations for FP8/NVFP4 inference (like `Self-Forcing-FP8`). Quantizing the model to FP8 doubles matrix multiplication throughput and halves the VRAM footprint without writing custom CUDA kernels from scratch. This allows the `[2, 2, 2]` setting to run as fast as the low-quality `[1, 1, 1]` setting.

### 3. GenRL for Advanced Motion and Aesthetic Tuning
When dropping Helios to the `[1, 1, 1]` speed-test setting, the quality becomes smeared and glitchy. GenRL is the missing piece for solving the visual quality problem.
*   **FlowGRPO Reinforcement Learning:** We will run the Helios-Distilled model through GenRL using built-in `videoalign_mq` (Motion Quality) and `hpsv3_general` (Aesthetics) reward functions. We can effectively fine-tune the model to look structurally coherent and drastically better even at the `[1, 1, 1]` step count.

By combining LightX2V's FP8 engine with GenRL's reward tuning and RIFE's optical flow interpolation, we will hit the holy grail: **24 FPS real-time generation that actually looks like the paper's original high-quality benchmarks.**
