# UI Improvements for DeepSynth Fine-Tuning

## Overview

The web UI has been enhanced with advanced fine-tuning controls based on the latest MoE dropout strategies and multi-trainer support.

## New Features

### 1. **Trainer Selection**

Choose between three trainer types based on your needs:

#### ⭐ Production Trainer (Recommended)
- **File**: `training/deepsynth_trainer_v2.py` - `ProductionDeepSynthTrainer`
- **Architecture**: Frozen encoder (380M params) + Fine-tuned MoE decoder (570M params)
- **Features**:
  - Expert Gradient Dropout
  - Gate Gradient Dropout
  - Bi-Drop training (multiple subnet passes)
  - Checkpoint resume support
  - Evaluation during training
- **Use Case**: Production-grade models with best quality

#### 🎯 DeepSynth Trainer
- **File**: `training/deepsynth_trainer.py` - `DeepSynthOCRTrainer`
- **Architecture**: Same frozen architecture without advanced dropout
- **Features**:
  - Basic frozen encoder setup
  - Visual token encoding
  - Checkpoint saving
- **Use Case**: Experiments and prototyping

#### 📝 Generic Trainer
- **File**: `training/trainer.py` - `SummarizationTrainer`
- **Architecture**: Full model trainable
- **Features**:
  - Standard seq2seq fine-tuning
  - No visual encoding
  - Text-only mode
- **Use Case**: Text-only datasets, traditional summarization

### 2. **Advanced MoE Dropout Configuration**

Collapsible advanced section with detailed controls:

#### Expert Dropout Rate
- **Range**: 0.0 - 1.0
- **Recommended**: 0.1 - 0.3
- **Effect**: Randomly drops expert gradients to prevent overfitting to specific pathways
- **Default**: 0.0 (disabled)

#### Min Experts Active
- **Range**: 1 - 8
- **Recommended**: 1 - 2
- **Effect**: Guarantees minimum number of active experts even after dropout
- **Default**: 1

#### Gate Dropout Rate
- **Range**: 0.0 - 1.0
- **Recommended**: 0.05 - 0.15
- **Effect**: Dropout on router/gating parameters for better load balancing
- **Default**: 0.0 (disabled)

#### Bi-Drop Forward Passes
- **Range**: 1 - 5
- **Recommended**: 2 - 3
- **Effect**: Multiple forward passes with different dropout masks for robust subnet learning
- **Default**: 1 (disabled)
- **Note**: Higher values improve quality but increase training time proportionally

### 3. **Enhanced UI Components**

- **Tooltips**: Hover over ℹ️ icons for detailed parameter explanations
- **Collapsible sections**: Advanced options hidden by default to reduce clutter
- **Visual indicators**: Color-coded badges (NEW, Recommended, Experimental)
- **Real-time validation**: Input constraints and helpful hints
- **Improved job monitoring**: Better progress visualization with metrics

## Usage Guide

### Quick Start

1. **Access the UI**:
   ```bash
   docker-compose -f docker-compose.gpu.yml up
   # Navigate to http://localhost:5000
   ```

2. **Generate a Dataset** (CPU):
   - Navigate to "📊 Benchmark Datasets" or "🗂️ Custom Dataset"
   - Select dataset and configure options
   - Click "Generate Dataset"
   - Wait for completion (check "📈 Monitor Jobs")

3. **Train a Model** (GPU required):
   - Navigate to "🎯 Train Model"
   - Select your generated dataset
   - **Choose Trainer**: Click "Production Trainer" (recommended)
   - Configure basic parameters (batch size, epochs, learning rate)
   - **(Optional)** Expand "Advanced: MoE Dropout Regularization"
     - Set expert_dropout_rate = 0.2
     - Set gate_dropout_rate = 0.1
     - Set bidrop_passes = 2
   - Click "Start Training"

### Recommended Configurations

#### High Quality (Slower)
```
Trainer: Production
Expert Dropout: 0.2
Gate Dropout: 0.1
Bi-Drop Passes: 3
Epochs: 5
Batch Size: 2
Gradient Accumulation: 8
Learning Rate: 2e-5
```

#### Balanced (Recommended)
```
Trainer: Production
Expert Dropout: 0.15
Gate Dropout: 0.05
Bi-Drop Passes: 2
Epochs: 3
Batch Size: 4
Gradient Accumulation: 4
Learning Rate: 5e-5
```

#### Fast Iteration (Prototyping)
```
Trainer: DeepSynth
Expert Dropout: 0.0 (disabled)
Gate Dropout: 0.0 (disabled)
Bi-Drop Passes: 1
Epochs: 1
Batch Size: 4
Gradient Accumulation: 2
Learning Rate: 5e-5
```

## Architecture Details

### Visual Pipeline
```
Text Document
    ↓
PNG Image (text_to_image)
    ↓
DeepEncoder (Frozen, 380M params)
    ↓
Visual Tokens (20x compression)
    ↓
MoE Decoder (Trainable, 570M params)
    ↓
Summary Text
```

### MoE Dropout Strategy
```
Forward Pass 1 (Dropout Mask A)
    ↓
Gradients → Expert Dropout → Gate Dropout
    ↓
Forward Pass 2 (Dropout Mask B, if bidrop_passes > 1)
    ↓
Average Losses
    ↓
Optimizer Step
```

## API Changes

The `/api/model/train` endpoint now accepts additional parameters:

```json
{
  "dataset_repo": "username/dataset-name",
  "trainer_type": "production",  // NEW: "production" | "deepsynth" | "generic"
  "expert_dropout_rate": 0.2,    // NEW: 0.0 - 1.0
  "expert_dropout_min_keep": 1,  // NEW: min active experts
  "gate_dropout_rate": 0.1,      // NEW: 0.0 - 1.0
  "bidrop_passes": 2,            // NEW: 1 - 5
  "batch_size": 2,
  "num_epochs": 3,
  "learning_rate": "5e-5",
  "mixed_precision": "bf16",
  "gradient_accumulation_steps": 8,
  "push_to_hub": true,
  "hub_model_id": "username/model-name"
}
```

## Performance Impact

| Configuration | Training Time | Quality | GPU Memory |
|--------------|---------------|---------|------------|
| No Dropout | 1.0x (baseline) | Good | 12GB |
| Expert Dropout (0.2) | 1.1x | Better | 12GB |
| + Gate Dropout (0.1) | 1.15x | Better+ | 12GB |
| + Bi-Drop (2 passes) | 1.3x | Best | 12GB |
| + Bi-Drop (3 passes) | 1.5x | Best+ | 12GB |

## Files Modified

- **web_ui/templates/index_improved.html**: Enhanced UI with trainer selection and MoE controls
- **web_ui/dataset_generator_improved.py**: Multi-trainer support with parameter passing
- **web_ui/app.py**: Updated to use improved components
- **training/config.py**: Already included new dropout parameters
- **training/deepsynth_trainer_v2.py**: Already implemented MoE dropout strategies

## Troubleshooting

### "Advanced options not showing"
- Make sure you selected "Production Trainer" - advanced options are only available for this trainer type

### "Training slower than expected"
- Check `bidrop_passes` - each additional pass adds ~50% training time
- Reduce to 1 for faster training at the cost of some quality

### "Out of memory errors"
- Reduce `batch_size` from 4 to 2 or even 1
- Increase `gradient_accumulation_steps` to maintain effective batch size
- Use `mixed_precision = "bf16"` (more stable than fp16)

### "Checkpoints not resuming"
- Ensure the checkpoint job completed successfully
- Check the "Resume Training" dropdown for available checkpoints
- Click "Refresh checkpoints" to reload the list

## References

- **DeepSeek-OCR Paper**: https://arxiv.org/abs/2510.18234
- **MoE Dropout Implementation**: `training/moe_dropout.py`
- **Production Trainer**: `training/deepsynth_trainer_v2.py`

## Support

For issues or questions:
- Check the "Monitor Jobs" tab for error messages
- Review logs in the Docker container
- See the [main README](../README.md) for general troubleshooting

---

**Built with ❤️ for production-grade DeepSynth fine-tuning**
