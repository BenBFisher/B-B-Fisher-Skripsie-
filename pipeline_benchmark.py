import torch
import psutil
import os
import time
import statistics
from datetime import datetime
from local_config_rt_detr_v2 import RTDetrV2Config
from local_modeling_rt_detr_v2 import RTDetrV2ForObjectDetection

def benchmark_pipeline_steps(device="cpu", image_size=640, num_warmup=10, num_runs=50):
    print(f"--- Benchmarking RT-DETR v2 Pipeline Steps on {device.upper()} ---")
    
    # 1. Setup Logging Directory and File
    log_dir = "resources_benchmark"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "pipeline_step_benchmark_log.txt")
    
    process = psutil.Process()
    
    # Maintain strict 80-class specification
    config = RTDetrV2Config(num_labels=80) 
    model = RTDetrV2ForObjectDetection(config).to(device)
    model.eval()

    #To test with local weights, uncomment the following lines and provide the correct path:
    # 1. Initialize the model using the strict 80-class specification
    config = RTDetrV2Config(num_labels=80) 
    model = RTDetrV2ForObjectDetection(config) 
    
    # 2. Load your custom trained weights
    print("Loading custom trained weights...")
    model.load_state_dict(torch.load("phase_1_weights_3107_coco5k.pth", map_location=torch.device('cpu')))
    
    # 3. Set to evaluation mode
    model.eval()
    #commented till here
    
    dummy_input = torch.randn(1, 3, image_size, image_size).to(device)
    
    # 2. Compute Parameter Count per Stage
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    enc_proj_params = sum(p.numel() for p in model.encoder_input_proj.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    
    bridge_params = (
        sum(p.numel() for p in model.decoder_input_proj.parameters()) +
        sum(p.numel() for p in model.enc_output.parameters()) +
        sum(p.numel() for p in model.enc_score_head.parameters()) +
        sum(p.numel() for p in model.enc_bbox_head.parameters())
    )
    
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    head_params = sum(p.numel() for p in model.class_embed.parameters()) + sum(p.numel() for p in model.bbox_embed.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    
    stage_names = [
        "1. Backbone",
        "2. Encoder Input Proj",
        "3. Hybrid Encoder (AIFI/FPN)",
        "4. Two-Stage Bridge",
        "5. RT-DETR Decoder",
        "6. Prediction Heads"
    ]
    
    stage_params = {
        "1. Backbone": backbone_params,
        "2. Encoder Input Proj": enc_proj_params,
        "3. Hybrid Encoder (AIFI/FPN)": encoder_params,
        "4. Two-Stage Bridge": bridge_params,
        "5. RT-DETR Decoder": decoder_params,
        "6. Prediction Heads": head_params
    }
    
    # Metrics collection data structures
    timings = {stage: [] for stage in stage_names}
    ram_deltas = {stage: [] for stage in stage_names}
    cpu_usage_per_stage = {stage: 0.0 for stage in stage_names}

    # 3. Warmup Runs
    print(f"Warming up pipeline ({num_warmup} iterations)...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)

    # 4. Step-by-Step Profile Loop (Latency & RAM Deltas)
    print(f"Measuring latency and RAM footprint over {num_runs} iterations...")
    
    with torch.no_grad():
        for _ in range(num_runs):
            # Step 1: Backbone
            m0 = process.memory_info().rss
            t0 = time.perf_counter()
            raw_feature_maps = model.backbone(dummy_input)
            t1 = time.perf_counter()
            m1 = process.memory_info().rss
            
            # Step 2: Encoder Input Projection
            projected_feature_maps = [
                proj(feat) for proj, feat in zip(model.encoder_input_proj, raw_feature_maps)
            ]
            t2 = time.perf_counter()
            m2 = process.memory_info().rss
            
            # Step 3: Hybrid Encoder
            encoded_features = model.encoder(projected_feature_maps)
            t3 = time.perf_counter()
            m3 = process.memory_info().rss
            
            # Step 4: Two-Stage Bridge
            sources = [model.decoder_input_proj[level](source) for level, source in enumerate(encoded_features)]
            source_flatten = []
            spatial_shapes_list = []
            spatial_shapes = torch.empty((len(sources), 2), device=device, dtype=torch.long)
            
            for level, source in enumerate(sources):
                height, width = source.shape[-2:]
                spatial_shapes[level, 0] = height
                spatial_shapes[level, 1] = width
                spatial_shapes_list.append((height, width))
                source_flatten.append(source.flatten(2).transpose(1, 2))
                
            source_flatten = torch.cat(source_flatten, 1)
            level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

            anchors, valid_mask = model.generate_anchors(spatial_shapes_list, device=device)
            memory = valid_mask.to(source_flatten.dtype) * source_flatten
            output_memory = model.enc_output(memory)

            enc_outputs_class = model.enc_score_head(output_memory)
            enc_outputs_coord_logits = model.enc_bbox_head(output_memory) + anchors

            _, topk_ind = torch.topk(enc_outputs_class.max(-1).values, model.config.num_queries, dim=1)
            reference_points_unact = enc_outputs_coord_logits.gather(
                dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_coord_logits.shape[-1])
            )
            init_reference_points = torch.sigmoid(reference_points_unact).detach()
            target = output_memory.gather(dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1])).detach()
            t4 = time.perf_counter()
            m4 = process.memory_info().rss
            
            # Step 5: Decoder
            decoder_outputs, final_reference_points = model.decoder(
                hidden_states=target,
                encoder_hidden_states=source_flatten,
                reference_points=init_reference_points,
                spatial_shapes=spatial_shapes,
                spatial_shapes_list=spatial_shapes_list,
                level_start_index=level_start_index,
                bbox_embed=model.bbox_embed
            )
            t5 = time.perf_counter()
            m5 = process.memory_info().rss
            
            # Step 6: Prediction Heads
            logits = model.class_embed[-1](decoder_outputs)
            t6 = time.perf_counter()
            m6 = process.memory_info().rss
            
            # Record Latencies (ms)
            timings["1. Backbone"].append((t1 - t0) * 1000)
            timings["2. Encoder Input Proj"].append((t2 - t1) * 1000)
            timings["3. Hybrid Encoder (AIFI/FPN)"].append((t3 - t2) * 1000)
            timings["4. Two-Stage Bridge"].append((t4 - t3) * 1000)
            timings["5. RT-DETR Decoder"].append((t5 - t4) * 1000)
            timings["6. Prediction Heads"].append((t6 - t5) * 1000)
            
            # Record RAM Footprint (MB)
            ram_deltas["1. Backbone"].append(max(0, (m1 - m0) / (1024 ** 2)))
            ram_deltas["2. Encoder Input Proj"].append(max(0, (m2 - m1) / (1024 ** 2)))
            ram_deltas["3. Hybrid Encoder (AIFI/FPN)"].append(max(0, (m3 - m2) / (1024 ** 2)))
            ram_deltas["4. Two-Stage Bridge"].append(max(0, (m4 - m3) / (1024 ** 2)))
            ram_deltas["5. RT-DETR Decoder"].append(max(0, (m5 - m4) / (1024 ** 2)))
            ram_deltas["6. Prediction Heads"].append(max(0, (m6 - m5) / (1024 ** 2)))

    # 5. Measure Isolated Stage CPU Usage
    print("Profiling stage-specific CPU utilization...")
    cpu_loop_runs = 100
    
    with torch.no_grad():
        # Backbone CPU %
        _ = process.cpu_percent(interval=None)
        for _ in range(cpu_loop_runs):
            _ = model.backbone(dummy_input)
        cpu_usage_per_stage["1. Backbone"] = process.cpu_percent(interval=None)

        # Encoder Proj CPU %
        _ = process.cpu_percent(interval=None)
        for _ in range(cpu_loop_runs):
            _ = [proj(feat) for proj, feat in zip(model.encoder_input_proj, raw_feature_maps)]
        cpu_usage_per_stage["2. Encoder Input Proj"] = process.cpu_percent(interval=None)

        # Hybrid Encoder CPU %
        _ = process.cpu_percent(interval=None)
        for _ in range(cpu_loop_runs):
            _ = model.encoder(projected_feature_maps)
        cpu_usage_per_stage["3. Hybrid Encoder (AIFI/FPN)"] = process.cpu_percent(interval=None)

        # Bridge CPU %
        _ = process.cpu_percent(interval=None)
        for _ in range(cpu_loop_runs):
            s_flat = torch.cat([model.decoder_input_proj[lvl](src).flatten(2).transpose(1, 2) for lvl, src in enumerate(encoded_features)], 1)
            anc, v_mask = model.generate_anchors(spatial_shapes_list, device=device)
            out_mem = model.enc_output(v_mask.to(s_flat.dtype) * s_flat)
            _ = model.enc_score_head(out_mem)
            _ = model.enc_bbox_head(out_mem) + anc
        cpu_usage_per_stage["4. Two-Stage Bridge"] = process.cpu_percent(interval=None)

        # Decoder CPU %
        _ = process.cpu_percent(interval=None)
        for _ in range(cpu_loop_runs):
            _ = model.decoder(
                hidden_states=target,
                encoder_hidden_states=source_flatten,
                reference_points=init_reference_points,
                spatial_shapes=spatial_shapes,
                spatial_shapes_list=spatial_shapes_list,
                level_start_index=level_start_index,
                bbox_embed=model.bbox_embed
            )
        cpu_usage_per_stage["5. RT-DETR Decoder"] = process.cpu_percent(interval=None)

        # Prediction Heads CPU %
        _ = process.cpu_percent(interval=None)
        for _ in range(cpu_loop_runs):
            _ = model.class_embed[-1](decoder_outputs)
        cpu_usage_per_stage["6. Prediction Heads"] = process.cpu_percent(interval=None)

    # 6. Aggregate Calculations
    median_stage_ms = {stage: statistics.median(times) for stage, times in timings.items()}
    avg_ram_mb = {stage: statistics.mean(deltas) for stage, deltas in ram_deltas.items()}
    
    total_pipeline_ms = sum(median_stage_ms.values())
    total_ram_alloc_mb = sum(avg_ram_mb.values()) if sum(avg_ram_mb.values()) > 0 else 1.0

    # 7. Format Detailed Table Report
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_lines = [
        f"[{timestamp}] Pipeline Step Resource Profile (Device: {device.upper()} | Image Size: {image_size}x{image_size})",
        f"Total Parameters: {total_params:,} | Classes Spec: {config.num_labels}",
        f"Total Pipeline Latency: {total_pipeline_ms:.2f} ms",
        f"{'='*105}",
        f"{'Pipeline Stage':<28} | {'Params (% Share)':<20} | {'Latency (% Share)':<20} | {'RAM Footprint':<18} | {'CPU Usage':<10}",
        f"{'='*105}"
    ]
    
    for stage_name in stage_names:
        p_count = stage_params[stage_name]
        p_pct = (p_count / total_params) * 100
        
        med_ms = median_stage_ms[stage_name]
        lat_pct = (med_ms / total_pipeline_ms) * 100
        
        ram_mb = avg_ram_mb[stage_name]
        ram_pct = (ram_mb / total_ram_alloc_mb) * 100 if total_ram_alloc_mb > 0 else 0.0
        
        cpu_pct = cpu_usage_per_stage[stage_name]
        
        report_lines.append(
            f"{stage_name:<28} | {p_count:<10,} ({p_pct:<4.1f}%) | {med_ms:<8.2f} ms ({lat_pct:<4.1f}%) | {ram_mb:<7.2f} MB ({ram_pct:<4.1f}%) | {cpu_pct:<6.1f}%"
        )
        
    report_lines.append(f"{'='*105}\n")
    report_text = "\n".join(report_lines)
    
    # 8. Output Results
    print("\n" + report_text)
    
    with open(log_file, "a") as f:
        f.write(report_text)
        
    print(f"Sub-component resource profile successfully logged to: {log_file}")

if __name__ == "__main__":
    benchmark_pipeline_steps(device="cpu")