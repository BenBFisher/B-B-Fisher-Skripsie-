# B-B-Fisher-Skripsie
Implementation of a RT-DETR (Real-Time Detection Transformer) onto edge computing. Repository


# RT-DETRv2 Edge Deployment: Heterogeneous Pipeline (Workstation $\to$ Raspberry Pi 4B $\to$ Raspberry Pi Pico 2)

This repository contains the complete implementation, model conversion toolchains, Ahead-of-Time (AOT) bare-metal compilation pipelines, and distributed serial runtime for deploying a **Real-Time Detection Transformer (RT-DETRv2)** onto resource-constrained embedded hardware. 

Developed as part of **Project (E) 448** at **Stellenbosch University**, the project investigates the feasibility of scaling down a high-capacity vision transformer across three hardware tiers, culminating in bare-metal execution on the **520 KB SRAM Raspberry Pi Pico 2 (RP2350)**.

---

## System Architecture & Pipeline Overview

The deployment methodology is decomposed into three distinct phases:

+--------------------------------------------------------------------------------------------------+
| PHASE 1: Desktop Reconstruction & Native Baseline                                                |
|   - Reconstructed PyTorch Graph (torch.nn primitives)                                            |
|   - Native Loss (Hungarian Matching + Focal Loss + GIoU)                                         |
|   - Full MS COCO 2017 Validation (AP50 = 0.646; 99.5% Reference Parity)                          |
+--------------------------------------------------------------------------------------------------+
|
v [Export: PyTorch -> ONNX]
+--------------------------------------------------------------------------------------------------+
| PHASE 2: Intermediate Edge Sandbox (Raspberry Pi 4B)                                             |
|   - Conversion: ONNX -> onnx2tf -> TF SavedModel -> TFLite Converter                             |
|   - 8-Bit Post-Training Quantization (Static PTQ Calibration)                                    |
|   - Layer-by-Layer Activation Memory Profiling: M_act = H * W * C * rho                          |
|   - TFLite Runtime Interpreter Memory Leak Identification (> 800 KB Dynamic Arena)               |
+--------------------------------------------------------------------------------------------------+
|
v [Aggressive Topological Downscaling]
+---------------------------------------------------------------------------------------------------+
| PHASE 3: Microcontroller Deployment & Distributed Pipeline (Raspberry Pi Pico 2)                  |
|   - Micro-Architecture: MobileNet Backbone, 32x32 / 160x160 Input, 2 Target Classes               |
|   - Apache TVM (microTVM) Ahead-of-Time (AOT) C Compilation (Statically Allocated Arena)          |    
|   - Distributed Pipeline: Pi 4B (Video / V4L2) <--- UART / USB Serial ---> Pico 2 (AOT Inference) |
|   - Bare-Metal RP2350 C Firmware: 2.11 FPS Sustained Roundtrip Throughput                         |
+---------------------------------------------------------------------------------------------------+


