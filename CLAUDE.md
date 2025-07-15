# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with the FlowFormer optical flow component.

## Component Overview

FlowFormer is a transformer-based architecture for optical flow estimation. This component implements the LatentCostFormer model that uses attention mechanisms to estimate motion between consecutive frames, serving as the optical flow backbone for the integrated fire detection system.

## Core Architecture

### FlowFormer LatentCostFormer Model
- **Encoder**: Twins-SVT transformer for feature extraction
- **Memory Encoder**: Cost volume encoding with latent space compression  
- **Memory Decoder**: Iterative flow refinement using transformer decoder
- **Key Innovation**: Latent cost volume reduces memory while preserving accuracy

### Model Configuration (configs/default.py)
```python
# Core architecture parameters
transformer = 'latentcostformer'
encoder_latent_dim = 256        # Twins encoder output dimension
cost_latent_dim = 128          # Compressed cost volume dimension  
decoder_depth = 12             # Transformer decoder layers
patch_size = 8                 # Feature patch size
```

## File Structure and Dependencies

### Core Implementation
- `train_FlowFormer.py` - Main training script with multi-stage pipeline
- `evaluate_FlowFormer.py` - Evaluation without tiling technique
- `evaluate_FlowFormer_tile.py` - Evaluation with memory-efficient tiling
- `visualize_flow.py` - Flow visualization utilities

### Model Architecture (`core/FlowFormer/`)
- `LatentCostFormer/transformer.py` - Main FlowFormer model class
- `LatentCostFormer/encoder.py` - Memory encoder for cost volume processing
- `LatentCostFormer/decoder.py` - Memory decoder for iterative refinement
- `LatentCostFormer/attention.py` - Multi-head attention mechanisms
- `encoders.py` - Feature encoders (Twins-SVT variants)

### Utilities (`core/`)
- `datasets.py` - Dataset loaders for FlyingChairs, Sintel, KITTI, etc.
- `loss.py` - Sequence loss function for multi-scale supervision
- `corr.py` - Correlation volume computation
- `utils/` - Flow visualization, augmentation, frame utilities

### Configuration (`configs/`)
- `default.py` - Base LatentCostFormer configuration
- `chairs.py`, `things.py`, `sintel.py`, `kitti.py` - Stage-specific configs
- `submission.py` - Benchmark submission configuration

## Common Development Commands

```bash
# Set working directory
cd FlowFormer_Official

# Environment setup
conda create --name flowformer
conda activate flowformer  
conda install pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.1 matplotlib tensorboard scipy opencv -c pytorch
pip install yacs loguru einops timm imageio

# Training pipeline (all stages)
./run_train.sh

# Individual training stages
python -u train_FlowFormer.py --name chairs --stage chairs --validation chairs
python -u train_FlowFormer.py --name things --stage things --validation sintel  
python -u train_FlowFormer.py --name sintel --stage sintel --validation sintel
python -u train_FlowFormer.py --name kitti --stage kitti --validation kitti

# Evaluation with tiling (recommended)
python evaluate_FlowFormer_tile.py --eval sintel_validation
python evaluate_FlowFormer_tile.py --eval kitti_validation --model checkpoints/things_kitti.pth

# Evaluation without tiling  
python evaluate_FlowFormer.py --dataset sintel

# Small model evaluation
python evaluate_FlowFormer_tile.py --eval sintel_validation --small

# Benchmark submission generation
python evaluate_FlowFormer_tile.py --eval sintel_submission
python evaluate_FlowFormer_tile.py --eval kitti_submission

# Visualization
python visualize_flow.py --eval_type sintel --keep_size
python visualize_flow.py --eval_type seq
```

## Training Workflow

### Multi-Stage Training Pipeline
1. **Stage 1 - Chairs**: Train on FlyingChairs (synthetic data)
2. **Stage 2 - Things**: Fine-tune on FlyingThings3D (more complex synthetic)  
3. **Stage 3 - Sintel**: Fine-tune on MPI-Sintel (real sequences)
4. **Stage 4 - KITTI**: Fine-tune on KITTI (automotive scenes)

### Training Configuration
- **Optimizer**: AdamW with OneCycleLR scheduler
- **Learning Rate**: 25e-5 canonical, with 1e-4 weight decay
- **Batch Size**: 8 (configurable)
- **Steps**: 120,000 training steps per stage
- **Loss**: Multi-scale sequence loss with exponential weights

### Configuration System (YACS)
```python
# Example configuration access
from configs.sintel import get_cfg
cfg = get_cfg()
cfg.batch_size = 4          # Adjust batch size
cfg.trainer.canonical_lr = 12.5e-5  # Adjust learning rate
```

## Model Architecture Details

### LatentCostFormer Components

**1. Memory Encoder** (`LatentCostFormer/encoder.py`):
- Processes image pairs to generate cost volumes
- Compresses 4D cost volume to latent tokens 
- Uses Twins-SVT encoder for feature extraction

**2. Memory Decoder** (`LatentCostFormer/decoder.py`):
- Iterative refinement using transformer decoder
- 12 decoder layers with self/cross attention
- Outputs multi-scale flow predictions

**3. Twins Encoder** (`encoders.py`):
- Twins-SVT-Large backbone for feature extraction
- Pretrained weights available (`pretrain=True`)
- Output dimension: 256 (encoder_latent_dim)

### Key Technical Features
- **Latent Cost Volume**: Reduces memory from O(H²W²) to O(HW)
- **Patch-based Processing**: 8x8 patches for efficient computation
- **Multi-Scale Loss**: Supervision at multiple resolutions
- **Memory Efficient**: Tiling technique for large images

## Dataset Integration

### Supported Datasets
- **FlyingChairs**: Synthetic training data
- **FlyingThings3D**: Complex synthetic scenes
- **MPI-Sintel**: Real movie sequences (clean/final)
- **KITTI**: Automotive stereo sequences  

### Dataset Structure
```
datasets/
├── Sintel/
│   ├── test/
│   └── training/
├── KITTI/  
│   ├── testing/
│   ├── training/
│   └── devkit/
├── FlyingChairs_release/
│   └── data/
└── FlyingThings3D/
    ├── frames_cleanpass/
    ├── frames_finalpass/
    └── optical_flow/
```

### Custom Dataset Integration
Implement `FlowDataset` subclass in `core/datasets.py`:
```python
class CustomFlowDataset(FlowDataset):
    def __init__(self, aug_params=None, sparse=False):
        super().__init__(aug_params, sparse)
        # Custom initialization
        
    def __getitem__(self, index):
        # Return (img1, img2, flow, valid_mask)
```

## Model Checkpoints and Performance

### Available Models (`checkpoints/`)
- `chairs.pth` - FlyingChairs trained model
- `things.pth` - FlyingThings3D trained model  
- `sintel.pth` - MPI-Sintel fine-tuned model
- `kitti.pth` - KITTI fine-tuned model
- `flowformer-small.pth` - Lightweight version
- `things_kitti.pth` - Combined model for KITTI training evaluation

### Performance Benchmarks
**Sintel (clean/final)**:
- With tiling: 0.94 / 2.33 EPE
- Without tiling: 1.01 / 2.40 EPE

**Small Model**:
- With tiling: 1.21 / 2.61 EPE  
- Without tiling: 1.32 / 2.68 EPE

## Integration with Fire Detection System

### Data Flow Integration
1. **Input**: Consecutive RGB frames from fire detection dataset
2. **Processing**: FlowFormer generates optical flow (.flo files)
3. **Output**: Flow fields for integration with RGB data
4. **Format**: Standard Middlebury .flo format (PIEH header + flow data)

### Custom Inference Function
```python
@torch.no_grad()
def inference_flow_512(model, dataloader, output_dir):
    """Generate 512x512 optical flow for fire detection integration"""
    for idx, (img1, img2) in enumerate(dataloader):
        img1, img2 = img1.cuda(), img2.cuda()
        
        # Resize to 512x512 for consistency
        img1 = F.interpolate(img1, size=(512, 512), mode='bilinear')
        img2 = F.interpolate(img2, size=(512, 512), mode='bilinear')
        
        # Apply padding for model requirements
        padder = InputPadder(img1.shape)
        img1_pad, img2_pad = padder.pad(img1, img2)
        
        # Inference
        flow_pad = model(img1_pad, img2_pad)[0]
        flow = padder.unpad(flow_pad)[0]
        
        # Save as .flo file
        flow_np = flow.permute(1, 2, 0).cpu().numpy()
        writeFlow(f"{output_dir}/frame_{idx:04d}.flo", flow_np)
```

### Memory Management
- **Tiling**: Use `evaluate_FlowFormer_tile.py` for large images when needed
- **Dynamic Batch Size**: Adjust batch_size based on runtime CUDA memory availability
- **Mixed Precision**: Automatic mixed precision supported for memory optimization

## Dependencies and Requirements

### Core Dependencies
- PyTorch 1.6.0+
- torchvision 0.7.0+
- CUDA toolkit 10.1+
- yacs (configuration management)
- loguru (logging)
- einops (tensor operations)  
- timm (latest version)
- imageio (image I/O)

### Optional Components
- **CUDA Correlation**: `alt_cuda_corr/` for performance optimization
  ```bash
  cd alt_cuda_corr
  python setup.py build_ext --inplace
  ```

## Troubleshooting

### Common Issues
- **timm Version**: Latest version is compatible with the current repository
- **CUDA Memory**: Handle dynamically - use tiling or adjust batch size based on actual memory errors
- **Model Loading**: Check checkpoint paths and model architecture matching
- **Dataset Paths**: Verify symbolic links to dataset directories

### Performance Optimization
- Enable CUDA correlation module for faster training
- Use mixed precision training (`torch.cuda.amp`)
- Apply tiling for memory-efficient inference when encountering memory constraints
- Adjust batch size dynamically based on available GPU memory
- Monitor actual memory usage rather than pre-setting conservative limits

### Integration Notes
- Flow output format: Standard .flo files (H×W×2 float32)
- Coordinate system: (u,v) = (x-displacement, y-displacement)  
- Range scaling: Flow values in pixels, scale appropriately for resized images
- Quality validation: Use forward-backward consistency checking