# 🎬 AI Dance Coach - Installation Guide for First-Time Users

**Welcome!** 🎉 This guide is written for **complete beginners**. We'll walk you through everything step-by-step.

---

## 📋 Table of Contents

1. [System Requirements](#-system-requirements)
2. [Before You Start](#-before-you-start)
3. [Installation Steps](#-installation-steps-follow-these-in-order)
4. [Verification](#-verify-everything-works)
5. [Troubleshooting](#-troubleshooting)
6. [Getting Help](#-getting-help)

---

## ✅ System Requirements

Check if your computer has these requirements:

### **Minimum Requirements**
- ✅ Windows 10/11, macOS 10.14+, or Linux (Ubuntu 18.04+)
- ✅ **Python 3.9** installed on your computer
- ✅ **5+ GB free disk space**
- ✅ **Internet connection** (for downloading packages)
- ✅ **Webcam** (for real-time feedback)

### **Recommended Requirements** (for best performance)
- ✅ **NVIDIA GPU** (like RTX, GTX series) - dramatically speeds up pose detection
- ✅ **NVIDIA CUDA drivers** updated (for GPU acceleration)
- ✅ **8+ GB RAM**
- ✅ **SSD** (faster file operations)

### **What If I Don't Have a GPU?**
✅ **No problem!** The software works on CPU too, just slower. You can still use it.

---

## 🔧 Before You Start

### **What is Python?**
Python is a programming language. Think of it like an interpreter that reads code instructions.

### **What is a Virtual Environment?**
A "sandbox" folder on your computer where we install packages just for this project. It keeps things isolated and organized.

### **Do I Need to Know How to Code?**
❌ **No!** Follow the copy-paste commands. We handle the complicated parts.

---

## 🚀 Installation Steps (Follow These in Order!)

### **Step 0️⃣: Download and Install Python 3.9**

Python is required first. Let's install it:

#### **Windows**
1. Go to: https://www.python.org/downloads/release/python-3913/
2. Scroll down and click: **"Windows installer (64-bit)"**
3. Run the downloaded file
4. ⚠️ **IMPORTANT:** Check the box that says **"Add Python to PATH"**
5. Click "Install Now"
6. Wait for installation to complete

#### **macOS**
1. Go to: https://www.python.org/downloads/release/python-3913/
2. Click: **"macOS 64-bit universal2 installer"**
3. Run the downloaded file and follow prompts
4. Python will be installed automatically

#### **Linux (Ubuntu/Debian)**
```bash
sudo apt update
sudo apt install python3.9 python3.9-venv python3.9-dev
```

**Verify Python installed:**
```bash
python --version
# Should show: Python 3.9.x
```

---

### **Step 1️⃣: Download This Repository**

You have two options:

#### **Option A: Using Git (Recommended if you have Git)**
```bash
git clone https://github.com/yourusername/AIDanceCoach.git
cd AIDanceCoach
```

#### **Option B: Using GitHub Downloads (If you don't have Git)**
1. Go to: https://github.com/yourusername/AIDanceCoach
2. Click green button: **"< > Code"**
3. Click: **"Download ZIP"**
4. Unzip the folder to a location like `C:\Users\YourName\Documents\AIDanceCoach`
5. Open a terminal/PowerShell in that folder

---

### **Step 2️⃣: Open a Terminal in Your Project Folder**

You need to open a command prompt/terminal in the `AIDanceCoach` folder.

#### **Windows (PowerShell)**
1. Open File Explorer
2. Navigate to where you downloaded `AIDanceCoach`
3. Click the address bar at the top
4. Type: `powershell`
5. Press Enter
6. You'll see a PowerShell window open

#### **macOS/Linux (Terminal)**
1. Open Terminal
2. Type: `cd /path/to/AIDanceCoach`
3. Press Enter

**You should see something like:**
```
C:\Users\YourName\AIDanceCoach>  (Windows)
or
~/AIDanceCoach $  (Mac/Linux)
```

---

### **Step 3️⃣: Create a Virtual Environment**

A virtual environment is like a "project folder" for Python packages.

#### **Windows (PowerShell)**
```powershell
python -m venv .venv
```

#### **macOS/Linux (Terminal)**
```bash
python3.9 -m venv venv
```

**This creates a folder called `.venv` or `venv` - don't worry about it, it's just a folder.**

---

### **Step 4️⃣: Activate the Virtual Environment**

This "turns on" your virtual environment.

#### **Windows (PowerShell)**
```powershell
.\.venv\Scripts\Activate.ps1
```

#### **macOS/Linux (Terminal)**
```bash
source venv/bin/activate
```

**Success looks like this:**
```
(.venv) C:\Users\YourName\AIDanceCoach>  (Windows)
or
(venv) ~/AIDanceCoach $  (Mac/Linux)
```

Notice the `(.venv)` or `(venv)` at the beginning - that means it's active!

---

### **Step 5️⃣: Upgrade pip (The Package Installer)**

This ensures you have the latest version of pip (which installs Python packages).

```bash
python -m pip install --upgrade pip setuptools wheel
```

**Wait for it to complete. You should see:**
```
Successfully installed pip-26.x.x setuptools-82.x.x wheel-0.46.x
```

---

### **Step 6️⃣: Check If You Have a GPU**

If you have an NVIDIA GPU, we can use it for faster processing.

```bash
nvidia-smi
```

**If you see GPU information, great!** Note down the `CUDA Version` (usually 12.1 or 11.8).

**If you get an error**, that's OK - you'll just use CPU instead.

---

### **Step 7️⃣: Install PyTorch (The Deep Learning Framework)**

This is the AI engine. The installation depends on whether you have a GPU.

#### **If you have an NVIDIA GPU (from Step 6):**

**For CUDA 12.1 or 12.7** (most common):
```bash
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
```

**For CUDA 11.8:**
```bash
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
```

⏳ **This will take 5-10 minutes - it's downloading ~2.5 GB. Don't close the terminal!**

#### **If you DON'T have an NVIDIA GPU (or prefer CPU):**
```bash
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2
```

**Verify it worked:**
```bash
python -c "import torch; print(torch.__version__); print('CUDA Available:', torch.cuda.is_available())"
```

You should see something like:
```
2.1.2+cu121
CUDA Available: True   (if GPU) or False (if CPU)
```

---

### **Step 8️⃣: Install OpenMIM (Package Manager for AI Models)**

```bash
pip install -U openmim
```

**Wait for completion.**

---

### **Step 9️⃣: Install MMEngine (AI Framework)**

```bash
mim install mmengine==0.10.5
```

**Wait for completion.**

---

### **Step 🔟: Install MMCV (Computer Vision Toolkit)**

This is the critical part. MMCV has special CUDA operators built in.

#### **If you have CUDA 12.1 or 12.7:**
```bash
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html
```

#### **If you have CUDA 11.8:**
```bash
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html
```

#### **If you don't have CUDA (CPU only):**
```bash
pip install mmcv==2.1.0
```

**This might say "Attempting uninstall: numpy" - that's expected!**

**Verify it worked:**
```bash
python -c "import mmcv; print('MMCV Version:', mmcv.__version__)"
```

---

### **Step 1️⃣1️⃣: Fix NumPy Version (IMPORTANT!)**

⚠️ **Some packages try to install NumPy 2.x, which breaks MMCV.**

Run this to ensure compatibility:

```bash
pip install "numpy<2.0.0" --force-reinstall --no-deps
```

**Verify:**
```bash
python -c "import numpy; print('NumPy Version:', numpy.__version__)"
```

Should show: `1.26.x`

---

### **Step 1️⃣2️⃣: Install Other AI Models**

```bash
pip install mmaction2==1.2.0 mmdet==3.3.0 --no-deps
```

**Note:** `mmpose` has compatibility issues on Windows, but you have MediaPipe instead (faster!).

---

### **Step 1️⃣3️⃣: Install Computer Vision & Media Libraries**

```bash
pip install mediapipe>=0.10.9 opencv-python>=4.8.0 opencv-contrib-python>=4.8.0
```

This installs:
- **MediaPipe** - Real-time pose detection
- **OpenCV** - Video processing

---

### **Step 1️⃣4️⃣: Install Video & Audio Processing**

```bash
pip install yt-dlp>=2024.3.10 librosa>=0.10.1 soundfile>=0.12.0 ffmpeg-python>=0.2.0
```

This installs:
- **yt-dlp** - Download YouTube videos
- **librosa** - Audio analysis
- **soundfile** - Audio file handling
- **ffmpeg-python** - Video manipulation

---

### **Step 1️⃣5️⃣: Install Hardware Optimization**

```bash
pip install onnxruntime-gpu>=1.16.3
```

(Use `onnxruntime` instead if you don't have a GPU)

---

### **Step 1️⃣6️⃣: Install Utilities**

```bash
pip install scipy>=1.11.0 fastdtw>=0.3.4 tqdm>=4.65.0 pyyaml>=6.0 colorama>=0.4.6
```

---

### **Step 1️⃣7️⃣: Verify Everything Works**

Now let's make sure everything installed correctly:

```bash
python verify_installation.py
```

**You should see lots of [OK] checkmarks.** If you see `[ERROR]`, check the troubleshooting section below.

**Expected output (partial):**
```
[OK] torch: 2.1.2+cu121
[OK] mmcv: 2.1.0
[OK] mediapipe: 0.10.32
[OK] PyTorch CUDA: CUDA 12.1, GPU available
```

---

## ✅ Verify Everything Works

### **Quick Test 1: Can I Import Packages?**

```bash
python -c "import torch, mmcv, mediapipe, cv2; print('✓ All packages imported!')"
```

### **Quick Test 2: Is My GPU Working?**

```bash
python -c "import torch; print('GPU Available:', torch.cuda.is_available()); print('GPU Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

### **Full Verification**

```bash
python verify_installation.py
```

Look for:
- ✅ All core packages showing version numbers
- ✅ GPU detected (if you have one)
- ✅ MMCV CUDA operations working

---

## 🐛 Troubleshooting

### **Issue: "Python command not found"**

**Solution:**
- Make sure Python is installed (Step 0)
- On Windows, add Python to PATH:
  - Type `python` in Windows search
  - Open "Edit environment variables for your account"
  - Find `PATH` and edit it
  - Add: `C:\Users\YourName\AppData\Local\Programs\Python\Python39`
  - Restart terminal

---

### **Issue: "No such file or directory: '.venv/Scripts/Activate.ps1'"**

**Solution:**
- Make sure you're in the `AIDanceCoach` folder
- Check that `.venv` folder exists
- If it doesn't, run Step 3 again

---

### **Issue: "CUDA Available: False" (but you have a GPU)**

**Solution:**
1. Check `nvidia-smi` works
2. Check CUDA drivers are installed
3. Reinstall PyTorch:
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
   ```

---

### **Issue: "ImportError: DLL load failed" (Windows)**

**Solution:**
- Install Visual C++ Redistributable:
  - https://aka.ms/vs/17/release/vc_redist.x64.exe
  - Run it
  - Restart terminal

---

### **Issue: "No module named 'mmpose'"**

**Solution:** This is expected on Windows. You have `mediapipe` which is better for real-time use!

---

### **Issue: Installation stuck/frozen**

**Solution:**
```bash
# Press Ctrl+C to stop
# Then try:
pip cache purge
pip install [package-name] --no-cache-dir
```

---

### **Issue: "numpy._core" errors**

**Solution:**
```bash
pip install "numpy<2.0.0" --force-reinstall --no-deps
```

---

### **My Installation Failed. How do I restart?**

**No problem!** Start fresh:

```bash
# 1. Delete the virtual environment
# Windows:
Remove-Item -Recurse -Force .venv

# Mac/Linux:
rm -rf venv

# 2. Start from Step 3 again
python -m venv .venv
# ... and continue from Step 4
```

---

## ❓ Getting Help

### **Where to Find Answers:**

1. **Read [INSTALLATION_STATUS.md](INSTALLATION_STATUS.md)** - Our detailed status report
2. **Check [README.md](README.md)** - Project overview and usage
3. **Search existing issues** - https://github.com/yourusername/AIDanceCoach/issues
4. **Open a new issue** - Include:
   - What step you got stuck on
   - Full error message
   - Your OS and Python version (`python --version`)
   - Your GPU info (`nvidia-smi`)

---

## 🎉 You Did It!

If you got here, congratulations! 🎊

Your AI Dance Coach environment is ready to use. Now:

1. **Read [README.md](README.md)** for how to use the software
2. **Run a first video** to see it in action
3. **Join the community** by opening issues or contributing

---

## 📚 Quick Reference

### **Common Commands**

**Activate virtual environment (always run first!):**
```bash
# Windows
.\.venv\Scripts\Activate.ps1

# Mac/Linux
source venv/bin/activate
```

**Deactivate when done:**
```bash
deactivate
```

**See all installed packages:**
```bash
pip list
```

**Update a package:**
```bash
pip install --upgrade [package-name]
```

**See my GPU:**
```bash
nvidia-smi
```

**Check Python version:**
```bash
python --version
```

---

## 💡 Tips & Tricks

✅ **Always activate the virtual environment first** - It's easy to forget!  
✅ **Keep terminal window open** - Some installs take a while  
✅ **Copy-paste the commands** - Typos can cause weird errors  
✅ **If something fails, try again** - Sometimes it's just a network hiccup  
✅ **Check GPU is working** - Verify `torch.cuda.is_available()` returns `True`  

---

## 🚀 Next Steps

Once installation is complete:

1. **Download Models:**
   ```bash
   mkdir -p models
   # Download RTMO (pose estimation model)
   # Instructions in README.md
   ```

2. **Run First Test:**
   ```bash
   python main.py --help
   ```

3. **Try with a YouTube Video:**
   ```bash
   python main.py --url "https://www.youtube.com/watch?v=XXXXX"
   ```

---

## 📖 Additional Reading

- [Python Virtual Environments Explained](https://docs.python.org/3/tutorial/venv.html)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [MMCV Documentation](https://mmcv.readthedocs.io/)
- [MediaPipe Pose Estimation](https://mediapipe.dev/)

---

## ❤️ Thank You!

Thanks for trying AI Dance Coach! We hope you enjoy building and using it. Happy coding! 🕺💃

---

**Questions?** Open an issue on GitHub!  
**Want to contribute?** Check [CONTRIBUTING.md](CONTRIBUTING.md)  
**Bug reports?** Include error message and system info  

**Happy Dancing! 🎬✨**
