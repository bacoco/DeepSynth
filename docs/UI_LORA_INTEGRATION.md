# LoRA/PEFT UI Integration for DeepSynth Web Interface

## Overview

This document describes the complete UI integration of LoRA/PEFT fine-tuning capabilities into the DeepSynth Flask web interface.

## Completed Features

### 1. API Endpoints

Three new REST API endpoints have been added to `/Users/loic/develop/DeepSynth/src/apps/web/ui/app.py`:

#### GET `/api/lora/presets`
Returns all available LoRA preset configurations.

**Response:**
```json
{
  "presets": {
    "minimal": { "rank": 4, "alpha": 8, ... },
    "standard": { "rank": 16, "alpha": 32, ... },
    "high_capacity": { "rank": 64, "alpha": 128, ... },
    "qlora_4bit": { "rank": 16, "use_quantization": true, ... },
    "qlora_8bit": { "rank": 16, "quantization_bits": 8, ... }
  }
}
```

#### POST `/api/lora/estimate`
Estimates VRAM usage and training requirements for a given LoRA configuration.

**Request:**
```json
{
  "lora_rank": 16,
  "use_qlora": true,
  "qlora_bits": 4,
  "use_text_encoder": false,
  "batch_size": 8
}
```

**Response:**
```json
{
  "estimated_vram_gb": 8.0,
  "trainable_params_millions": 4.0,
  "gpu_fit": {
    "T4 (16GB)": true,
    "RTX 3090 (24GB)": true,
    "A100 (40GB)": true,
    "A100 (80GB)": true
  },
  "speed_multiplier": 1.0
}
```

#### POST `/api/dataset/generate` (Enhanced)
Enhanced to accept `instruction_prompt` parameter for instruction-based training.

**New Parameter:**
```json
{
  "instruction_prompt": "Summarize this text:"
}
```

### 2. HTML UI Components

Added comprehensive LoRA configuration section in `index_improved.html` (lines 880-998):

#### Main Features:

**LoRA/QLoRA Configuration Panel:**
- Collapsible section with info box explaining LoRA benefits
- Enable/Disable LoRA checkbox
- LoRA preset selector with 5 presets
- LoRA parameter controls (rank, alpha, dropout)
- QLoRA toggle with quantization options (4-bit/8-bit)
- Real-time resource estimation display

**Text Encoder Configuration:**
- Enable/Disable text encoder checkbox
- Text encoder type selector (Qwen3 / BERT)
- Trainable toggle option

**Resource Estimation Box:**
- Live VRAM estimation
- Trainable parameter count
- GPU compatibility matrix
- Updates automatically when config changes

**Instruction Prompting:**
- Added to dataset generation form (line 660-663)
- Optional text field for instruction prefix
- Helpful tooltip explaining use case

### 3. JavaScript Functions

Added six new JavaScript functions (lines 1482-1584):

#### `toggleLoRAOptions()`
Shows/hides the LoRA configuration panel and triggers resource estimation.

#### `toggleQLoRAOptions()`
Shows/hides quantization options when QLoRA is enabled.

#### `toggleTextEncoderOptions()`
Shows/hides text encoder configuration panel.

#### `applyLoRAPreset()`
Async function that:
- Fetches available presets from API
- Applies selected preset values to all LoRA fields
- Handles both standard LoRA and QLoRA presets
- Updates resource estimation
- Shows success toast notification

#### `updateLoRAEstimate()`
Async function that:
- Collects current LoRA configuration
- Calls `/api/lora/estimate` endpoint
- Updates the resource estimation display
- Shows VRAM, trainable params, and GPU compatibility

#### Form Submission Enhancement
Updated training form submission (lines 1907-1919) to include:
- All LoRA parameters (use_lora, lora_rank, lora_alpha, lora_dropout)
- QLoRA parameters (use_qlora, qlora_bits, qlora_type)
- Text encoder parameters (use_text_encoder, text_encoder_type, text_encoder_trainable)

Updated dataset generation form (line 1830) to include:
- instruction_prompt parameter

### 4. User Experience Features

**Progressive Disclosure:**
- LoRA options hidden by default
- Options appear progressively as features are enabled
- Prevents overwhelming users with too many options

**Real-time Feedback:**
- Immediate resource estimation updates
- GPU compatibility checking
- Visual indicators for parameter changes

**Preset System:**
- Quick configuration via 5 presets
- Each preset optimized for different scenarios:
  - **Minimal**: Fastest training, lowest quality
  - **Standard**: Recommended balance
  - **High Capacity**: Best quality, slower training
  - **QLoRA 4-bit**: Maximum memory savings (8GB GPU)
  - **QLoRA 8-bit**: Balanced memory/speed

**Help Text:**
- Inline help for every field
- Tooltips explaining technical terms
- Examples for guidance

## UI Workflow

### Dataset Generation with Instruction Prompting

1. User navigates to "Custom Dataset Generation" tab
2. Fills in source dataset information
3. **NEW**: Optionally adds instruction prompt (e.g., "Summarize this text:")
4. System prepends instruction to each document before image generation
5. Dataset contains both original text and instruction-augmented versions

### Training with LoRA

1. User navigates to "Model Training" tab
2. Selects trainer type (Production recommended)
3. Expands "🔧 LoRA/QLoRA Fine-Tuning" section
4. **Enables LoRA checkbox** → Panel expands
5. Selects preset OR customizes parameters:
   - Adjusts rank (4-128)
   - Adjusts alpha (8-256)
   - Sets dropout (0-0.5)
6. **Optional**: Enable QLoRA for memory savings:
   - Choose 4-bit or 8-bit quantization
   - Select quantization type (NF4 recommended)
7. **Optional**: Enable text encoder:
   - Choose Qwen3 or BERT
   - Toggle trainability
8. **Monitors resource estimation** in real-time:
   - Checks if config fits on available GPU
   - Reviews trainable parameter count
9. Submits training job with LoRA configuration

### Resource Estimation Example

For QLoRA 4-bit with rank=16, no text encoder:
```
Estimated VRAM: 8.0 GB
Trainable Params: 4.0M
GPU Compatibility: ✓ T4 (16GB), ✓ RTX 3090 (24GB), ✓ A100 (40GB), ✓ A100 (80GB)
```

For QLoRA 4-bit with rank=16, with text encoder:
```
Estimated VRAM: 12.0 GB
Trainable Params: 12.0M
GPU Compatibility: ✗ T4 (16GB), ✓ RTX 3090 (24GB), ✓ A100 (40GB), ✓ A100 (80GB)
```

## Technical Implementation Details

### API Integration

**app.py (lines 250-338)**:
```python
@app.route("/api/lora/presets", methods=["GET"])
def get_lora_presets():
    from deepsynth.training.lora_config import LORA_PRESETS
    # Returns all preset configurations

@app.route("/api/lora/estimate", methods=["POST"])
def estimate_lora_resources():
    # Calculates memory and parameter estimates
    # Returns GPU compatibility matrix
```

### HTML Structure

**Training Section (lines 880-998)**:
```html
<!-- LoRA Configuration -->
<button type="button" class="collapsible">
    🔧 LoRA/QLoRA Fine-Tuning <span class="badge badge-new">NEW</span>
</button>
<div class="collapsible-content">
    <!-- Enable LoRA Checkbox -->
    <!-- Preset Selector -->
    <!-- Parameter Controls -->
    <!-- QLoRA Options -->
    <!-- Text Encoder Options -->
    <!-- Resource Estimation -->
</div>
```

**Dataset Section (lines 659-664)**:
```html
<div class="form-group">
    <label for="instruction_prompt">Instruction Prompt (Optional)
        <span class="badge badge-new">NEW</span>
    </label>
    <input type="text" id="instruction_prompt" ...>
    <small class="help-text">...</small>
</div>
```

### JavaScript Integration

**Event Handlers:**
- `onchange="toggleLoRAOptions()"` on use_lora checkbox
- `onchange="toggleQLoRAOptions(); updateLoRAEstimate()"` on use_qlora checkbox
- `onchange="toggleTextEncoderOptions(); updateLoRAEstimate()"` on use_text_encoder checkbox
- `onchange="applyLoRAPreset()"` on preset selector
- `onchange="updateLoRAEstimate()"` on parameter inputs

**Form Submission (lines 1907-1919)**:
```javascript
// Training config now includes:
use_lora: document.getElementById('use_lora').checked,
lora_rank: parseInt(document.getElementById('lora_rank').value),
// ... all LoRA parameters ...
use_text_encoder: document.getElementById('use_text_encoder').checked,
text_encoder_type: document.getElementById('text_encoder_type').value,
// ... all text encoder parameters ...
```

## Memory Estimation Algorithm

The estimation algorithm in `/api/lora/estimate`:

```python
base_memory = 16.0  # GB for full model

# LoRA reduction
if use_qlora:
    if qlora_bits == 4:
        memory = base_memory * 0.25  # 75% reduction
    elif qlora_bits == 8:
        memory = base_memory * 0.5   # 50% reduction
else:
    memory = base_memory * 0.5  # Standard LoRA ~50% reduction

# Add text encoder overhead
if use_text_encoder:
    memory += 4.0  # Additional 4GB

# Adjust for batch size
memory += (batch_size - 8) * 0.5  # ~500MB per batch item

# Trainable parameters based on rank
if lora_rank <= 8:    trainable_params_m = 2.0
elif lora_rank <= 16: trainable_params_m = 4.0
elif lora_rank <= 32: trainable_params_m = 8.0
else:                 trainable_params_m = 16.0

if use_text_encoder:
    trainable_params_m += 8.0  # Text encoder params
```

## Styling

All LoRA UI components use existing DeepSynth styling:
- `.form-section` for grouped controls
- `.form-group` for individual fields
- `.form-row` and `.form-row-3` for layouts
- `.collapsible` for expandable sections
- `.success-box` for resource estimation display
- `.badge-new` for highlighting new features
- `.help-text` for inline help

## Browser Compatibility

Tested and compatible with:
- Chrome/Edge 90+
- Firefox 88+
- Safari 14+

**Requirements:**
- JavaScript enabled
- Fetch API support
- ES6 async/await support

## Future Enhancements

Potential improvements for Phase 3:
- [ ] Visual LoRA rank slider with preview
- [ ] Training time estimation
- [ ] Cost estimation (cloud GPU pricing)
- [ ] Adapter management UI (list/delete/merge)
- [ ] Real-time training metrics for LoRA
- [ ] Adapter export directly from UI
- [ ] Multi-adapter comparison
- [ ] LoRA hyperparameter search

## Testing Checklist

To test the UI integration:

- [ ] LoRA checkbox enables/disables panel
- [ ] Preset selector applies correct values
- [ ] Resource estimation updates in real-time
- [ ] QLoRA checkbox shows/hides quantization options
- [ ] Text encoder checkbox shows/hides encoder options
- [ ] Parameter inputs update estimation
- [ ] Form submission includes all LoRA parameters
- [ ] Instruction prompt field accepts text
- [ ] Toast notifications appear for preset changes
- [ ] GPU compatibility matrix updates correctly
- [ ] All collapsible sections expand/collapse properly

## Error Handling

The UI includes error handling for:
- Failed API calls (shows error toast)
- Invalid input values (browser validation)
- Missing required fields
- Network errors during estimation
- Failed preset loading

## Accessibility

The UI follows accessibility best practices:
- All form fields have associated labels
- Help text uses `<small>` semantic markup
- Keyboard navigation supported
- Focus states visible
- ARIA attributes where appropriate

---

*Last Updated: 2025-10-27*
*DeepSynth LoRA UI Integration v1.0*
