# 🧍 Image to 3D Model Pipeline

Generate realistic 3D human body models using SMPL-X on Google Colab.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Colab](https://img.shields.io/badge/Google%20Colab-Ready-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [What This Project Does](#-what-this-project-does)
- [Technologies Used](#-technologies-used)
- [Prerequisites](#-prerequisites)
- [Setup Guide](#-setup-guide)
- [Running on Google Colab](#-running-on-google-colab)
- [Project Structure](#-project-structure)
- [Understanding the Pipeline](#-understanding-the-pipeline)
- [SMPL-X Deep Dive](#-smpl-x-deep-dive)
- [Parameters Reference](#-parameters-reference)
- [Code Explanation](#-code-explanation-cell-by-cell)
- [Troubleshooting](#-troubleshooting)
- [FAQ](#-faq--interview-questions)
- [Future Improvements](#-future-improvements)
- [Resources](#-resources)

---

## 🎯 Overview

This project creates a pipeline to generate **3D human body meshes** from parametric body models. It uses **SMPL-X** (Skinned Multi-Person Linear model with eXpressive hands and face) to create customizable 3D human bodies.

### Why SMPL-X?

SMPL-X is the industry standard for human body modeling, used by:
- **Meta/Facebook** - Avatar creation
- **Google** - Motion capture research
- **Disney** - Character animation
- **Academic research** - Thousands of papers

---

## 🔄 What This Project Does

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  SMPL-X Model   │────▶│  Generate Mesh  │────▶│  Export/Render  │
│  (from Drive)   │     │  (10,475 verts) │     │  (OBJ/PLY/PNG)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
   Load .npz files      Apply shape params      Save to Drive
   (NEUTRAL/MALE/       (betas, pose)           View in Blender
    FEMALE)
```

### Pipeline Steps:

1. **Mount Google Drive** - Access your SMPL-X model files
2. **Install Dependencies** - PyTorch, SMPL-X, PyRender, Trimesh, etc.
3. **Setup OSMesa** - Enable headless rendering on Colab
4. **Load SMPL-X Model** - Initialize the body model
5. **Generate Mesh** - Create 3D body with 10,475 vertices
6. **Customize Shape** - Modify body proportions with beta parameters
7. **Render** - Create images with PyRender (realistic lighting)
8. **Export** - Save as OBJ/PLY for external software

---

## 🛠️ Technologies Used

### Core Libraries

| Library | Version | Purpose | Why We Use It |
|---------|---------|---------|---------------|
| **PyTorch** | 2.0+ | Deep learning framework | Required by SMPL-X for tensor ops |
| **SMPL-X** | 0.1.28 | Parametric body model | The core model that generates bodies |
| **PyTorch3D** | 0.7+ | 3D deep learning | Mesh operations, differentiable rendering |
| **Trimesh** | 3.x | Mesh I/O | Load/save OBJ, PLY, compute normals |
| **PyRender** | 0.1.45 | 3D rendering | Create realistic images with lighting |
| **Open3D** | 0.17+ | 3D processing | Point cloud, mesh visualization |
| **NumPy** | 1.x | Arrays | Convert tensors, numerical operations |
| **Matplotlib** | 3.x | Plotting | Simple 3D visualization |

### System Dependencies (Linux/Colab)

| Package | Purpose |
|---------|---------|
| **libosmesa6-dev** | Off-screen Mesa rendering (no display needed) |
| **freeglut3-dev** | OpenGL utility toolkit |
| **libgl1-mesa-dev** | Mesa OpenGL implementation |
| **libglu1-mesa-dev** | OpenGL Utility Library |

### Why These Specific Libraries?

```
PyTorch ──────────▶ SMPL-X needs tensors and autograd
                    (could use TensorFlow, but SMPL-X is PyTorch-native)

SMPL-X ───────────▶ Industry standard, well-documented, active development
                    (alternatives: SCAPE, GHUM - less supported)

PyRender ─────────▶ Simple API, works headless with OSMesa
                    (alternatives: Blender Python - overkill for this)

Trimesh ──────────▶ Lightweight, handles all mesh formats
                    (alternatives: PyMesh - harder to install)
```

---

## 📦 Prerequisites

### 1. Google Account
- Required for Google Colab and Google Drive

### 2. SMPL-X Model Files
Download from official website (requires registration):

🔗 **https://smpl-x.is.tue.mpg.de/**

**Files needed:**
```
SMPLX_NEUTRAL.npz  (~400 MB) - Gender-neutral body
SMPLX_MALE.npz     (~400 MB) - Male body model
SMPLX_FEMALE.npz   (~400 MB) - Female body model
```

### 3. Google Drive Setup
Upload SMPL-X models to this structure:
```
My Drive/
└── Computer_Models/
    └── models/
        └── smplx/
            ├── SMPLX_NEUTRAL.npz
            ├── SMPLX_MALE.npz
            └── SMPLX_FEMALE.npz
```

---

## 🚀 Setup Guide

### Step 1: Register for SMPL-X

1. Go to https://smpl-x.is.tue.mpg.de/
2. Click "Register" (requires academic email or justification)
3. Accept the license agreement
4. Download "SMPL-X v1.1 (NPZ+PKL, 830MB)"

### Step 2: Extract and Upload

```bash
# After downloading smplx_v1_1.zip
unzip smplx_v1_1.zip
# Upload the 'models' folder to Google Drive
```

### Step 3: Verify Drive Structure

Your Google Drive should look like:
```
📁 My Drive
└── 📁 Computer_Models
    └── 📁 models
        └── 📁 smplx
            ├── 📄 SMPLX_NEUTRAL.npz (397 MB)
            ├── 📄 SMPLX_MALE.npz (397 MB)
            └── 📄 SMPLX_FEMALE.npz (397 MB)
```

---

## 🖥️ Running on Google Colab

### Quick Start (5 minutes)

1. **Upload notebook** to Colab:
   - Go to https://colab.research.google.com/
   - File → Upload notebook → Select `Image_to_3D_model.ipynb`

2. **Enable GPU** (recommended):
   - Runtime → Change runtime type → **T4 GPU**

3. **Run all cells**:
   - Runtime → Run all (Ctrl+F9)

4. **Authorize Drive** when prompted

### Cell Execution Timeline

| Cell | Section | What It Does | Expected Time |
|------|---------|--------------|---------------|
| 00 | Mount Drive | Connect to Google Drive | ~10 sec |
| 0 | Setup | Set paths, environment vars | <1 sec |
| 1 | Install | pip install all packages | ~3-5 min |
| 2 | OSMesa | apt-get install OpenGL libs | ~30 sec |
| 3 | Clone | git clone smplx repo | ~10 sec |
| 4 | Import | Import Python libraries | ~5 sec |
| 5 | Load Model | Initialize SMPL-X | ~10 sec |
| 6 | Generate | Create T-pose mesh | <1 sec |
| 7 | Matplotlib | 3D scatter plot visualization | ~5 sec |
| 8 | PyRender | Offscreen realistic render | ~10 sec |
| 9 | Save | Export OBJ and PLY files | <1 sec |
| 10 | Custom | Generate custom body shape | ~5 sec |

**Total: ~5-7 minutes** (first run, subsequent runs faster)

### Expected Outputs

```
/content/outputs/
├── meshes/
│   ├── body_neutral.obj     # Default T-pose (10,475 vertices)
│   ├── body_neutral.ply     # Same mesh in PLY format
│   └── body_custom.obj      # Custom shape (taller/thinner)
└── renders/
    ├── body_views.png       # Matplotlib front/side/back
    └── body_pyrender.png    # PyRender realistic render
```

---

## 📁 Project Structure

```
Image to 3D model/
├── 📓 Image_to_3D_model.ipynb   # Main Colab notebook
├── 📄 README.md                  # This documentation
├── 📁 models_smplx_v1_1/         # Local models (optional)
│   └── models/
│       └── smplx/
│           ├── SMPLX_NEUTRAL.npz
│           ├── SMPLX_MALE.npz
│           └── SMPLX_FEMALE.npz
└── 📁 outputs/                   # Generated files
    ├── meshes/
    │   └── *.obj, *.ply
    └── renders/
        └── *.png
```


---

## 🔬 Understanding the Pipeline

### How 3D Body Generation Works

```
                    SMPL-X MODEL
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
    Shape (β)        Pose (θ)      Expression (ψ)
    10 params        55 joints      10 params
         │               │               │
         └───────────────┼───────────────┘
                         ▼
              Linear Blend Skinning
                         │
                         ▼
              10,475 Vertices (x,y,z)
                         │
                         ▼
               20,908 Triangular Faces
                         │
                         ▼
                  3D MESH OUTPUT
```

### The Math Behind It

**SMPL-X Function:**
```
M(β, θ, ψ) = W(T(β, θ, ψ), J(β), θ, W)
```

Where:
- `β` = Shape parameters (body proportions)
- `θ` = Pose parameters (joint rotations)
- `ψ` = Expression parameters (facial expressions)
- `T` = Template mesh deformation
- `J` = Joint locations
- `W` = Blend weights (how much each joint affects each vertex)

### Step-by-Step Process

1. **Template Mesh**: Start with average human body (T-pose)
2. **Shape Blend**: Add shape variations based on β parameters
3. **Pose Blend**: Correct for pose-dependent deformations
4. **Expression Blend**: Add facial expressions
5. **Skinning**: Rotate body parts around joints
6. **Output**: Final 3D mesh coordinates

---

## 🧠 SMPL-X Deep Dive

### What is SMPL-X?

**SMPL-X** = **S**kinned **M**ulti-**P**erson **L**inear model with e**X**pressive hands and face

It's a **parametric body model** - meaning you can control the body shape and pose with a small set of numbers (parameters), and the model generates the full 3D mesh.

### SMPL Family Evolution

```
SMPL (2015)      SMPL+H (2017)     SMPL-X (2019)
   │                  │                 │
   ▼                  ▼                 ▼
Body only         + Hands           + Face + Hands
6,890 verts      10,475 verts      10,475 verts
23 joints        52 joints         55 joints
```

### Model Statistics

| Property | Value |
|----------|-------|
| Vertices | 10,475 |
| Faces | 20,908 |
| Joints | 55 (22 body + 30 hand + 3 face) |
| Shape params (β) | 10-300 (usually 10) |
| Pose params (θ) | 55 joints × 3 = 165 |
| Expression params (ψ) | 10 |
| Hand pose params | 12 (6 per hand, PCA) |

### Body Parts Breakdown

```
Head (3 joints)
├── Jaw
├── Left Eye
└── Right Eye

Torso (5 joints)
├── Pelvis (root)
├── Spine 1-3
└── Neck

Arms (6 joints each)
├── Shoulder
├── Elbow
└── Wrist
    └── Hand (15 joints)
        ├── Thumb (3)
        ├── Index (3)
        ├── Middle (3)
        ├── Ring (3)
        └── Pinky (3)

Legs (5 joints each)
├── Hip
├── Knee
├── Ankle
├── Foot
└── Toes
```

---

## 📊 Parameters Reference

### Shape Parameters (Betas)

Control body proportions. First 10 are most important:

| Beta | Approximate Effect |
|------|-------------------|
| β₀ | Overall body size / height |
| β₁ | Weight / body mass |
| β₂ | Shoulder width |
| β₃ | Hip width |
| β₄ | Arm length |
| β₅ | Leg length |
| β₆ | Torso length |
| β₇ | Chest size |
| β₈ | Waist size |
| β₉ | Neck thickness |

**Value ranges:** Typically -3 to +3 (standard deviations from mean)

```python
# Example: Tall and thin body
betas = torch.zeros([1, 10])
betas[0, 0] = 2.0   # Taller
betas[0, 1] = -1.5  # Thinner
```

### Pose Parameters (Theta)

Control joint rotations in **axis-angle** format (3 values per joint):

| Joint Index | Body Part |
|-------------|-----------|
| 0 | Pelvis (global rotation) |
| 1-3 | Spine (lower to upper) |
| 4 | Neck |
| 5 | Head |
| 6 | Left shoulder |
| 7 | Left elbow |
| 8 | Left wrist |
| 9-11 | Right arm |
| 12-14 | Left leg |
| 15-17 | Right leg |
| ... | Hands and face |

**Value ranges:** Typically -π to +π radians

```python
# Example: Raise right arm
body_pose = torch.zeros([1, 63])  # 21 body joints × 3
body_pose[0, 47] = -1.5  # Right shoulder Z rotation
```

### Gender Options

| Gender | Description | Use Case |
|--------|-------------|----------|
| `neutral` | Average of male/female | Default, unisex |
| `male` | Male body shape | Male characters |
| `female` | Female body shape | Female characters |

---

## 📝 Code Explanation (Cell by Cell)

### Cell 00: Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive', timeout_ms=60000)
```

**What it does:**
- Connects Colab to your Google Drive
- Makes your files accessible at `/content/drive/MyDrive/`
- `timeout_ms=60000` gives 60 seconds for authorization

**Why we need it:**
- SMPL-X model files are large (~400MB each)
- Storing them in Drive avoids re-uploading every session

### Cell 0: Setup Environment

```python
os.environ["XDG_RUNTIME_DIR"] = "/tmp/runtime-root"
os.environ["RUNLEVEL"] = "3"
```

**What it does:**
- Sets Linux environment variables for headless rendering
- Creates directories for output files

**Why we need it:**
- Colab runs without a display (headless)
- OpenGL needs these variables to work without a screen

### Cell 1: Install Dependencies

```python
!pip install torch smplx trimesh pyrender open3d
```

**What each package does:**
- `torch`: Tensor operations, GPU acceleration
- `smplx`: The body model library
- `trimesh`: Mesh I/O (read/write OBJ, PLY)
- `pyrender`: 3D rendering with lighting
- `open3d`: 3D visualization and processing

### Cell 2: Setup OSMesa

```python
!apt-get install -y libosmesa6-dev
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
```

**What it does:**
- Installs Mesa 3D library for off-screen rendering
- Tells PyOpenGL to use OSMesa backend

**Why we need it:**
- Colab has no display/monitor
- OSMesa allows OpenGL rendering to memory instead of screen

### Cell 5: Load SMPL-X Model

```python
model = smplx.create(
    model_path=SMPLX_FOLDER,
    model_type='smplx',
    gender='neutral',
    num_betas=10,
    use_pca=True,
    num_pca_comps=6,
    ext='npz'
)
```

**Parameters explained:**
- `model_path`: Where the .npz files are
- `model_type`: 'smpl', 'smplh', or 'smplx'
- `gender`: 'neutral', 'male', or 'female'
- `num_betas`: How many shape parameters (10 is standard)
- `use_pca`: Use PCA for hand poses (reduces parameters)
- `num_pca_comps`: Number of PCA components for hands
- `ext`: File extension ('npz' or 'pkl')

### Cell 6: Generate Mesh

```python
output = model(return_verts=True)
vertices = output.vertices.detach().cpu().numpy()[0]
faces = model.faces
```

**What it does:**
- Calls the model with default parameters (T-pose)
- Gets vertices as numpy array (10,475 × 3)
- Gets face indices (20,908 × 3)

**Output structure:**
```python
output.vertices  # Shape: [batch, 10475, 3]
output.joints    # Shape: [batch, 55, 3]
output.full_pose # Shape: [batch, 165]
```

### Cell 8: PyRender Visualization

```python
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import pyrender

scene = pyrender.Scene()
mesh_pyrender = pyrender.Mesh.from_trimesh(mesh)
scene.add(mesh_pyrender)

r = pyrender.OffscreenRenderer(640, 480)
color, depth = r.render(scene)
```

**What it does:**
- Creates a 3D scene with the body mesh
- Adds camera and lights
- Renders to an image (no display needed)

**Key concepts:**
- `OffscreenRenderer`: Renders to memory, not screen
- `color`: RGB image (H × W × 3)
- `depth`: Depth map (H × W)

---

## 🔧 Troubleshooting

### Common Errors and Solutions

#### 1. "SMPL-X model not found"

```
FileNotFoundError: SMPLX_NEUTRAL.npz not found
```

**Solution:**
- Check your Drive path: `/content/drive/MyDrive/Computer_Models/models/smplx/`
- Ensure files are uploaded and named correctly
- Run the Drive mount cell again

#### 2. "OSMesa not found"

```
ImportError: OSMesa not found
```

**Solution:**
```python
!apt-get update
!apt-get install -y libosmesa6-dev
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
```

#### 3. "CUDA out of memory"

```
RuntimeError: CUDA out of memory
```

**Solution:**
```python
# Use CPU instead
device = 'cpu'
model = model.to(device)

# Or reduce batch size
batch_size = 1
```

#### 4. "PyTorch3D installation failed"

```
error: command 'gcc' failed
```

**Solution:**
```python
# Install from pre-built wheel
!pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1120/download.html
```

#### 5. "Drive mount timeout"

```
TimeoutError: Mount timed out
```

**Solution:**
- Check internet connection
- Try unmounting and remounting:
```python
drive.flush_and_unmount()
drive.mount('/content/drive')
```

#### 6. "Invalid shape for betas"

```
RuntimeError: shape '[1, 10]' is invalid for input of size 20
```

**Solution:**
```python
# Ensure betas has correct shape
betas = torch.zeros([1, 10], device=device)  # [batch_size, num_betas]
```

---

## ❓ FAQ / Interview Questions

### Basic Questions

**Q: What is SMPL-X?**
> A: SMPL-X is a parametric 3D body model that represents the human body as a function of shape and pose parameters. It generates a mesh with 10,475 vertices and captures body, hands, and facial expressions.

**Q: What's the difference between SMPL and SMPL-X?**
> A: SMPL has only body (6,890 vertices, 23 joints). SMPL-X adds articulated hands (30 joints) and expressive face (3 joints), totaling 10,475 vertices and 55 joints.

**Q: Why use parametric models instead of raw 3D scans?**
> A: Parametric models are:
> - Compact (few parameters vs millions of vertices)
> - Controllable (change shape/pose independently)
> - Consistent topology (same vertex order for all bodies)
> - Animatable (smooth joint rotations)

**Q: What are beta parameters?**
> A: Betas (β) control body shape - things like height, weight, shoulder width. They represent coefficients of a PCA basis learned from thousands of 3D body scans.

### Intermediate Questions

**Q: How does Linear Blend Skinning (LBS) work?**
> A: LBS computes each vertex position as a weighted sum of transformations from nearby joints:
> ```
> v' = Σ w_i * T_i * v
> ```
> Where w_i are blend weights and T_i are joint transformations.

**Q: Why do we need OSMesa?**
> A: OSMesa (Off-Screen Mesa) allows OpenGL rendering without a physical display. Colab runs on headless servers, so we can't use regular OpenGL that expects a screen.

**Q: What's the difference between body_pose and global_orient?**
> A:
> - `global_orient`: 3 values, rotates entire body in world space
> - `body_pose`: 63 values (21 joints × 3), rotates individual body parts relative to parent joint

**Q: How is the face modeled in SMPL-X?**
> A: SMPL-X uses a linear expression model with 10 parameters (ψ). These control facial expressions by deforming face vertices around the jaw, eyes, and mouth areas.

### Advanced Questions

**Q: What is PCA and why use it for hands?**
> A: PCA (Principal Component Analysis) reduces dimensionality. Instead of 45 parameters for each hand (15 joints × 3), we use 6 PCA components that capture the most common hand poses. This prevents unrealistic hand configurations.

**Q: How are joint locations determined?**
> A: Joint locations J(β) are computed as a linear combination of vertices:
> ```
> J = J_regressor @ vertices
> ```
> The J_regressor is a sparse matrix learned during model training.

**Q: What's the difference between NPZ and PKL model files?**
> A: Both contain the same model weights. NPZ is NumPy's compressed format (faster to load in Python). PKL is Python pickle (slightly larger, same data).

**Q: How would you animate this model?**
> A: Change pose parameters over time:
> ```python
> for frame in range(100):
>     theta = base_pose + frame * delta_pose
>     output = model(body_pose=theta)
>     save_mesh(output, f"frame_{frame}.obj")
> ```

**Q: What are pose-dependent blend shapes?**
> A: Corrective deformations that fix artifacts from LBS. When you bend an elbow, pure LBS creates a "candy wrapper" effect. Pose blend shapes add learned corrections to make it look realistic.

---

## 🚀 Future Improvements

### Not Yet Implemented

- [ ] **Image to Pose**: Use HMR/PARE to estimate pose from photo
- [ ] **Hand Pose Control**: Full articulated finger control
- [ ] **Facial Expressions**: Blend shape controls for emotions
- [ ] **Texture Mapping**: UV coordinates for skin/clothing textures
- [ ] **Animation Export**: FBX/GLTF for game engines
- [ ] **Batch Processing**: Generate multiple bodies at once
- [ ] **Clothing Simulation**: Add garments to body
- [ ] **Collision Detection**: Prevent self-intersections

### Potential Extensions

```
Current Pipeline:
  Parameters → SMPL-X → Mesh → Export

Future Pipeline:
  Image → Pose Estimation → SMPL-X → Texture → Animation → Game Engine
                ↑
            HMR/PARE/
            PyMAF
```

---

## 📚 Resources

### Official Links

- [SMPL-X Website](https://smpl-x.is.tue.mpg.de/) - Model download
- [SMPL-X Paper](https://arxiv.org/abs/1904.05866) - Original research
- [GitHub: vchoutas/smplx](https://github.com/vchoutas/smplx) - Source code
- [PyTorch3D](https://pytorch3d.org/) - 3D deep learning

### Tutorials

- [SMPL-X Tutorial](https://smpl-x.is.tue.mpg.de/tutorial) - Official guide
- [Google Colab Basics](https://colab.research.google.com/notebooks/intro.ipynb)
- [PyRender Documentation](https://pyrender.readthedocs.io/)

### Related Projects

- [HMR](https://github.com/akanazawa/hmr) - Human Mesh Recovery from images
- [PARE](https://github.com/mkocabas/PARE) - Part Attention Regressor
- [PyMAF](https://github.com/HongwenZhang/PyMAF) - 3D pose from images
- [VIBE](https://github.com/mkocabas/VIBE) - Video body estimation

### Papers to Read

1. **SMPL** (2015): "SMPL: A Skinned Multi-Person Linear Model"
2. **SMPL-X** (2019): "Expressive Body Capture: 3D Hands, Face, and Body from a Single Image"
3. **MANO** (2017): "Embodied Hands: Modeling and Capturing Hands and Bodies Together"
4. **FLAME** (2017): "Learning a model of facial shape and expression from 4D scans"

---

## 📄 License

This project uses:
- **SMPL-X Model**: Academic/research license from MPI
- **Code**: MIT License

⚠️ SMPL-X models cannot be used for commercial purposes without permission from Max Planck Institute.

---

## 🙏 Acknowledgments

- Max Planck Institute for Intelligent Systems (SMPL-X)
- Facebook AI Research (PyTorch3D)
- Google Colaboratory (free GPU access)

---

*Last updated: January 2025*

