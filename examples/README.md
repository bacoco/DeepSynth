# 🎯 DeepSynth Examples

This directory contains example scripts and test files to help you understand and use DeepSynth.

## 🚀 Getting Started Examples

### Basic Usage Examples

- **[instruction_prompting_example.py](instruction_prompting_example.py)** - Example of instruction prompting with DeepSynth
- **[train_with_optimized_trainer.py](train_with_optimized_trainer.py)** - Example of training with optimized trainer

## 🧪 Test Scripts

### Integration Tests

- **[test_lora_integration.py](test_lora_integration.py)** - Test LoRA integration functionality
- **[test_nq_streaming_minimal.py](test_nq_streaming_minimal.py)** - Minimal test for Natural Questions streaming

## 🏃‍♂️ Running the Examples

### Prerequisites
```bash
cd ~/repos/DeepSynth
source .envrc  # Load environment variables
```

### Run Instruction Prompting Example
```bash
python examples/instruction_prompting_example.py
```

### Run LoRA Integration Test
```bash
python examples/test_lora_integration.py
```

### Run Training Example
```bash
python examples/train_with_optimized_trainer.py
```

### Run Streaming Test
```bash
python examples/test_nq_streaming_minimal.py
```

## 📚 What Each Example Demonstrates

| Example | What it shows | Use case |
|---------|---------------|----------|
| `instruction_prompting_example.py` | How to use instruction prompting | Custom prompting strategies |
| `train_with_optimized_trainer.py` | Optimized training setup | Production training |
| `test_lora_integration.py` | LoRA fine-tuning | Parameter-efficient training |
| `test_nq_streaming_minimal.py` | Streaming data processing | Large dataset handling |

## 🎯 Expected Outputs

### Successful Runs Should Show:
- ✅ Environment loaded correctly
- ✅ Models and datasets accessible
- ✅ Processing completed without errors
- ✅ Results saved to appropriate directories

### Common Issues:
- ❌ Missing HF_TOKEN in .env
- ❌ CUDA not available (for GPU examples)
- ❌ Insufficient disk space
- ❌ Network connectivity issues

## 🔧 Customization

You can modify these examples for your own use cases:

1. **Change model parameters** in the configuration
2. **Adjust dataset paths** and sizes
3. **Modify training hyperparameters**
4. **Add your own evaluation metrics**

## 📞 Need Help?

If examples don't work as expected:

1. Check the **[docs/QUICKSTART.md](../docs/QUICKSTART.md)** guide
2. Verify your **[.env](../.env.example)** configuration
3. Review **[docs/TROUBLESHOOTING.md](../docs/TROUBLESHOOTING.md)** (if available)

---

*Examples directory - Last updated: November 2024*