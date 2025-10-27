"""
Example Usage: VLA Model with Sensor Encoder Integration

This script demonstrates how to use the new QwenVLAWithSensor model
that combines vision-language features with OCT/FPI sensor data.

Key Features Demonstrated:
1. Model initialization with different fusion strategies
2. Data preparation for sensor input
3. Forward pass with multi-modal inputs
4. Training loop integration
5. Comparison with original model
"""

import torch
import numpy as np
from pathlib import Path

# Import the new sensor-integrated models
from model_with_sensor import (
    SensorEncoder,
    QwenActionExpertWithSensor,
    QwenVLAWithSensor,
    Not_freeze_QwenVLAWithSensor
)

# =====================================
# Example 1: Basic Inference with Frozen VL Backbone
# =====================================
def example_1_basic_inference():
    """
    Example of using QwenVLAWithSensor for inference
    with frozen VL backbone and trainable sensor encoder
    """
    print("\n" + "="*80)
    print("Example 1: Basic Inference with Sensor Data")
    print("="*80)

    # Initialize model with sensor encoder
    model = QwenVLAWithSensor(
        vl_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        action_dim=7,
        horizon=8,
        hidden_dim=1024,
        sensor_enabled=True,
        sensor_input_channels=1026,  # 1 force + 1025 A-scan
        sensor_temporal_length=650,   # 1 second at 650Hz
        sensor_output_dim=3072,       # Match VL dimension
        fusion_strategy='concat'      # Options: 'concat', 'cross_attention', 'gated', 'none'
    )

    model.eval()

    # Prepare dummy inputs
    batch_size = 2
    text_inputs = [
        "Pick up the object and place it in the box",
        "Grasp the tool and insert it into the hole"
    ]
    image_inputs = [
        ["path/to/left_view.jpg", "path/to/oak_view.jpg"],
        ["path/to/left_view.jpg", "path/to/oak_view.jpg"]
    ]
    z_chunk = torch.randn(batch_size, 8, 7)  # (B, horizon, action_dim)

    # Prepare sensor data: (B, T, C) where T=650, C=1026
    sensor_data = torch.randn(batch_size, 650, 1026)
    # In practice, this comes from OCT_FPI_Encoder/Sensor_reciver.py:
    # sensor_data = torch.from_numpy(np.concatenate([force_data[:, None], mmode_image_data], axis=-1))

    # Forward pass
    with torch.no_grad():
        pred_actions, delta = model(
            text_inputs=text_inputs,
            image_inputs=image_inputs,
            z_chunk=z_chunk,
            sensor_data=sensor_data,
            cache=True  # Use VL feature caching
        )

    print(f"✅ Prediction complete!")
    print(f"   Predicted actions shape: {pred_actions.shape}")  # (B, 8, 7)
    print(f"   Delta shape: {delta.shape}")  # (B, 8, 7)

    return pred_actions


# =====================================
# Example 2: Training with LoRA Fine-tuning
# =====================================
def example_2_training_with_lora():
    """
    Example of training with LoRA fine-tuning of VL backbone
    and end-to-end training of sensor encoder
    """
    print("\n" + "="*80)
    print("Example 2: Training with LoRA Fine-tuning")
    print("="*80)

    # Initialize trainable model with LoRA
    model = Not_freeze_QwenVLAWithSensor(
        vl_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        action_dim=7,
        horizon=8,
        hidden_dim=1024,
        finetune_vl="lora",  # Options: "none", "lora", "full"
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        sensor_enabled=True,
        sensor_input_channels=1026,
        sensor_temporal_length=650,
        sensor_output_dim=3072,
        fusion_strategy='cross_attention'  # Use cross-attention fusion
    )

    model.train()

    # Setup optimizer with different learning rates
    vl_params = []
    sensor_params = []
    action_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'vl_model' in name and 'lora' in name:
                vl_params.append(param)
            elif 'sensor_encoder' in name:
                sensor_params.append(param)
            elif 'action_expert' in name:
                action_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': vl_params, 'lr': 1e-5, 'name': 'vl_lora'},
        {'params': sensor_params, 'lr': 5e-4, 'name': 'sensor_encoder'},
        {'params': action_params, 'lr': 5e-4, 'name': 'action_expert'}
    ], weight_decay=0.01)

    # Training loop
    num_epochs = 3
    for epoch in range(num_epochs):
        # Dummy batch
        batch_size = 4
        text_inputs = ["Task instruction"] * batch_size
        image_inputs = [["view1.jpg", "view2.jpg"]] * batch_size
        z_chunk = torch.randn(batch_size, 8, 7)
        sensor_data = torch.randn(batch_size, 650, 1026)
        target_actions = torch.randn(batch_size, 8, 7)

        # Forward pass
        pred_actions, delta = model(
            text_inputs=text_inputs,
            image_inputs=image_inputs,
            z_chunk=z_chunk,
            sensor_data=sensor_data
        )

        # Compute loss
        loss = torch.nn.functional.mse_loss(pred_actions, target_actions)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

    print("✅ Training complete!")


# =====================================
# Example 3: Comparing Different Fusion Strategies
# =====================================
def example_3_fusion_strategies():
    """
    Compare different sensor-VL fusion strategies
    """
    print("\n" + "="*80)
    print("Example 3: Comparing Fusion Strategies")
    print("="*80)

    strategies = ['concat', 'cross_attention', 'gated', 'none']

    batch_size = 2
    text_inputs = ["Pick up the object"] * batch_size
    image_inputs = [["view.jpg"]] * batch_size
    z_chunk = torch.randn(batch_size, 8, 7)
    sensor_data = torch.randn(batch_size, 650, 1026)

    results = {}

    for strategy in strategies:
        print(f"\n🔹 Testing fusion strategy: {strategy}")

        model = QwenVLAWithSensor(
            vl_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
            action_dim=7,
            horizon=8,
            sensor_enabled=(strategy != 'none'),
            fusion_strategy=strategy
        )
        model.eval()

        with torch.no_grad():
            pred_actions, _ = model(
                text_inputs=text_inputs,
                image_inputs=image_inputs,
                z_chunk=z_chunk,
                sensor_data=sensor_data if strategy != 'none' else None
            )

        results[strategy] = pred_actions
        print(f"   ✅ Output shape: {pred_actions.shape}")

    # Compare results
    print("\n📊 Comparison of fusion strategies:")
    for strategy, pred in results.items():
        print(f"   {strategy}: mean={pred.mean().item():.4f}, std={pred.std().item():.4f}")


# =====================================
# Example 4: Integrating with Real Sensor Data
# =====================================
def example_4_real_sensor_integration():
    """
    Example of integrating with real OCT/FPI sensor data
    from Sensor_reciver.py
    """
    print("\n" + "="*80)
    print("Example 4: Real Sensor Data Integration")
    print("="*80)

    # Simulate real sensor data processing
    # In practice, this comes from OCT_FPI_Encoder/Sensor_reciver.py

    # Load saved sensor data (example)
    # sensor_file = "saved_data_1234567890.npz"
    # data = np.load(sensor_file)
    # forces = data['forces']  # (N,)
    # alines = data['alines']  # (N, 1025)

    # For this example, create dummy data
    num_samples = 2000  # Multiple seconds of data
    forces = np.random.randn(num_samples, 1).astype(np.float32)
    alines = np.random.randn(num_samples, 1025).astype(np.float32)

    # Combine force and A-scan data
    sensor_raw = np.concatenate([forces, alines], axis=-1)  # (N, 1026)

    # Extract 1-second sliding window (650 samples)
    def extract_window(data, window_size=650):
        """Extract the most recent window"""
        if len(data) < window_size:
            # Pad if not enough data
            padding = np.zeros((window_size - len(data), data.shape[1]))
            return np.vstack([padding, data])
        else:
            return data[-window_size:]

    sensor_window = extract_window(sensor_raw, 650)  # (650, 1026)

    # Prepare for model input: add batch dimension
    sensor_data = torch.from_numpy(sensor_window).unsqueeze(0)  # (1, 650, 1026)

    print(f"📊 Sensor data prepared:")
    print(f"   Raw data shape: {sensor_raw.shape}")
    print(f"   Window shape: {sensor_window.shape}")
    print(f"   Model input shape: {sensor_data.shape}")

    # Initialize model
    model = QwenVLAWithSensor(
        vl_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        sensor_enabled=True,
        fusion_strategy='concat'
    )
    model.eval()

    # Prepare other inputs
    text_inputs = ["Insert the probe into the tissue"]
    image_inputs = [["left_camera.jpg", "oak_camera.jpg"]]
    z_chunk = torch.randn(1, 8, 7)

    # Forward pass
    with torch.no_grad():
        pred_actions, _ = model(
            text_inputs=text_inputs,
            image_inputs=image_inputs,
            z_chunk=z_chunk,
            sensor_data=sensor_data
        )

    print(f"✅ Predicted actions: {pred_actions.shape}")


# =====================================
# Example 5: Model Parameter Analysis
# =====================================
def example_5_parameter_analysis():
    """
    Analyze trainable parameters in different configurations
    """
    print("\n" + "="*80)
    print("Example 5: Parameter Analysis")
    print("="*80)

    configs = [
        ("Frozen VL + Sensor", {
            "finetune_vl": "none",
            "sensor_enabled": True
        }),
        ("LoRA VL + Sensor", {
            "finetune_vl": "lora",
            "sensor_enabled": True
        }),
        ("Full VL + Sensor", {
            "finetune_vl": "full",
            "sensor_enabled": True,
            "unfreeze_last_n": 2
        }),
    ]

    for name, config in configs:
        print(f"\n🔹 Configuration: {name}")

        model = Not_freeze_QwenVLAWithSensor(
            vl_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
            **config
        )

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        vl_trainable = sum(p.numel() for n, p in model.named_parameters()
                          if 'vl_model' in n and p.requires_grad)
        sensor_trainable = sum(p.numel() for n, p in model.named_parameters()
                              if 'sensor_encoder' in n and p.requires_grad)
        action_trainable = sum(p.numel() for n, p in model.named_parameters()
                              if 'action_expert' in n and p.requires_grad)

        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        print(f"   └─ VL model: {vl_trainable:,}")
        print(f"   └─ Sensor encoder: {sensor_trainable:,}")
        print(f"   └─ Action expert: {action_trainable:,}")


# =====================================
# Example 6: Migration from Original Model
# =====================================
def example_6_migration_guide():
    """
    Guide for migrating from the original model to sensor-integrated model
    """
    print("\n" + "="*80)
    print("Example 6: Migration Guide")
    print("="*80)

    print("""
    MIGRATION FROM ORIGINAL MODEL:

    1. Original Model (model.py):
       ```python
       from model import QwenVLAForAction

       model = QwenVLAForAction(
           vl_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
           action_dim=7,
           horizon=8
       )

       pred_actions, delta = model(
           text_inputs=text_inputs,
           image_inputs=image_inputs,
           z_chunk=z_chunk
       )
       ```

    2. New Model WITHOUT Sensor (backward compatible):
       ```python
       from model_with_sensor import QwenVLAWithSensor

       model = QwenVLAWithSensor(
           vl_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
           action_dim=7,
           horizon=8,
           sensor_enabled=False  # Disable sensor encoder
       )

       pred_actions, delta = model(
           text_inputs=text_inputs,
           image_inputs=image_inputs,
           z_chunk=z_chunk,
           sensor_data=None  # No sensor data
       )
       ```

    3. New Model WITH Sensor:
       ```python
       from model_with_sensor import QwenVLAWithSensor

       model = QwenVLAWithSensor(
           vl_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
           action_dim=7,
           horizon=8,
           sensor_enabled=True,  # Enable sensor encoder
           sensor_input_channels=1026,
           sensor_temporal_length=650,
           fusion_strategy='concat'  # Choose fusion strategy
       )

       # Prepare sensor data from Sensor_reciver.py
       sensor_data = torch.randn(batch_size, 650, 1026)

       pred_actions, delta = model(
           text_inputs=text_inputs,
           image_inputs=image_inputs,
           z_chunk=z_chunk,
           sensor_data=sensor_data  # Add sensor data
       )
       ```

    KEY CHANGES:
    - sensor_enabled: Set to True to enable sensor encoder
    - sensor_data: Pass sensor data tensor (B, 650, 1026)
    - fusion_strategy: Choose how to combine VL and sensor features
      - 'concat': Simple concatenation (fastest)
      - 'cross_attention': Cross-attention fusion (most expressive)
      - 'gated': Learned gating (balanced)
      - 'none': VL only (backward compatible)

    TRAINING SCRIPT CHANGES (5st_VLA_TRAIN_VL_Lora.py):
    1. Replace import:
       ```python
       # OLD
       from model import Not_freeze_QwenVLAForAction

       # NEW
       from model_with_sensor import Not_freeze_QwenVLAWithSensor
       ```

    2. Add sensor data to dataset collate_fn
    3. Pass sensor_data to model forward call
    4. Adjust learning rates for sensor encoder parameters

    BENEFITS:
    - Multi-modal fusion of vision, language, and tactile sensing
    - Better performance on contact-rich tasks
    - Maintains backward compatibility
    - Flexible fusion strategies
    """)


# =====================================
# Main Execution
# =====================================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("VLA Model with Sensor Encoder - Usage Examples")
    print("="*80)

    # Run examples (comment out as needed)
    # example_1_basic_inference()
    # example_2_training_with_lora()
    # example_3_fusion_strategies()
    # example_4_real_sensor_integration()
    # example_5_parameter_analysis()
    example_6_migration_guide()

    print("\n" + "="*80)
    print("✅ All examples completed!")
    print("="*80)
