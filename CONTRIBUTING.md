# 🤝 Contributing to AI Dance Coach

Welcome, contributor! 🎉 We're thrilled you want to help make AI Dance Coach better.

---

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Setting Up for Development](#setting-up-for-development)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Submitting a Pull Request](#submitting-a-pull-request)
- [Reporting Issues](#reporting-issues)
- [Style Guide](#style-guide)

---

## 📜 Code of Conduct

We're committed to providing a welcoming and inclusive experience. Please:

- ✅ Be respectful and constructive
- ✅ Assume good intentions
- ✅ Give credit where it's due
- ✅ Focus on the code, not the person

Any harassment or discriminatory behavior will not be tolerated.

---

## 🚀 Getting Started

### **1. Fork the Repository**

Go to: https://github.com/yourusername/AIDanceCoach
Click: **Fork** (top-right button)

This creates your own copy of the project.

### **2. Clone Your Fork**

```bash
# Clone YOUR fork (not the original)
git clone https://github.com/YOURUSERNAME/AIDanceCoach.git
cd AIDanceCoach

# Add the original repo as "upstream" (to stay updated)
git remote add upstream https://github.com/yourusername/AIDanceCoach.git

# Verify
git remote -v
# Should show:
# origin    https://github.com/YOURUSERNAME/AIDanceCoach.git (fetch)
# origin    https://github.com/YOURUSERNAME/AIDanceCoach.git (push)
# upstream  https://github.com/yourusername/AIDanceCoach.git (fetch)
# upstream  https://github.com/yourusername/AIDanceCoach.git (push)
```

### **3. Create a Feature Branch**

Always create a branch for your changes:

```bash
# Update your fork with latest code
git fetch upstream
git checkout main
git merge upstream/main

# Create a new branch for your feature
git checkout -b feature/my-awesome-feature

# Or for a bug fix:
git checkout -b fix/issue-123-bug-description
```

**Branch naming convention:**
- `feature/feature-name` for new features
- `fix/issue-description` for bug fixes
- `docs/change-description` for documentation
- `test/test-description` for tests

---

## 🛠️ Setting Up for Development

### **Step 1: Install Development Dependencies**

Follow [FIRST_TIME_SETUP.md](FIRST_TIME_SETUP.md) for the main installation.

Then install development tools:

```bash
# Activate your virtual environment first!
# Windows:
.\.venv\Scripts\Activate.ps1
# Mac/Linux:
source venv/bin/activate

# Install dev dependencies
pip install pytest pytest-cov black flake8 isort mypy

# Install the package in editable mode
pip install -e .
```

### **Step 2: Verify Development Setup**

```bash
# Run tests
pytest tests/

# Check code quality
black --check src/
flake8 src/
```

---

## ✏️ Making Changes

### **Development Workflow**

```bash
# 1. Make sure you're on your feature branch
git branch
# Should show: * feature/my-awesome-feature

# 2. Make your changes
# Edit files in your editor

# 3. Test your changes
pytest tests/

# 4. Follow code style
black src/
isort src/
flake8 src/

# 5. Stage your changes
git add .

# 6. Commit with clear message
git commit -m "Add feature X: description of what you did"

# 7. Push to your fork
git push origin feature/my-awesome-feature
```

### **File Structure to Know**

```
src/
├── ingestion/          → Video downloading & processing
├── segmentation/       → Dance step segmentation
├── pose/              → Pose estimation & tracking
├── alignment/         → DTW scoring & alignment
└── ui/               → Visual feedback display

tests/                 → Mirror structure of src/
```

### **Editing Guidelines**

- **Python Style**: Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- **Docstrings**: Use Google-style docstrings
- **Comments**: Explain the "why", not the "what"
- **Type Hints**: Use type annotations where possible

**Example function:**
```python
def process_video(video_path: str, output_dir: str) -> bool:
    """
    Process a video file and save frames.
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save processed frames
        
    Returns:
        True if processing succeeded, False otherwise
        
    Raises:
        FileNotFoundError: If video_path doesn't exist
        IOError: If output_dir cannot be written to
    """
    # Implementation here
    pass
```

---

## 🧪 Testing

### **Running Tests**

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/pose/test_tracker.py

# Run with coverage
pytest --cov=src tests/

# Run with verbose output
pytest -v tests/
```

### **Writing Tests**

Tests go in `tests/` (mirroring `src/` structure).

```python
# Example: tests/pose/test_tracker.py

import pytest
from src.pose.tracker import BlazePoseTracker

class TestBlazePoseTracker:
    """Tests for BlazePoseTracker"""
    
    def setup_method(self):
        """Setup for each test"""
        self.tracker = BlazePoseTracker()
    
    def test_initialization(self):
        """Test tracker initializes correctly"""
        assert self.tracker is not None
    
    def test_invalid_input_raises_error(self):
        """Test that invalid input raises an error"""
        with pytest.raises(ValueError):
            self.tracker.process(None)
```

### **Test Coverage Goal**

Aim for at least:
- ✅ 70% code coverage for new features
- ✅ 80% coverage for critical modules (pose, alignment)

---

## 📤 Submitting a Pull Request

### **Before You Submit**

```bash
# 1. Update your branch with latest upstream code
git fetch upstream
git rebase upstream/main

# 2. Fix any conflicts
# (if prompted, resolve in your editor)

# 3. Run full test suite
pytest tests/
black src/
flake8 src/

# 4. Push your final changes
git push origin feature/my-awesome-feature
```

### **Create the Pull Request**

1. Go to: https://github.com/yourusername/AIDanceCoach
2. Click: **Pull requests** tab
3. Click: **New Pull Request**
4. Set:
   - **Base:** yourusername/AIDanceCoach (main)
   - **Compare:** your-fork/AIDanceCoach (your-branch)
5. Click: **Create Pull Request**

### **Fill in the PR Template**

```markdown
## Description
Brief description of what this PR does.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Related Issue
Closes #123

## How to Test
Steps to test this change:
1. Do this
2. Then that
3. Verify this

## Checklist
- [ ] Code follows style guidelines
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] No new warnings generated
```

### **PR Review Process**

1. Maintainers will review your code
2. They might request changes
3. Make updates and push again (same branch)
4. Once approved, your PR will be merged!

---

## 🐛 Reporting Issues

### **Found a Bug?**

1. Check if issue already exists: https://github.com/yourusername/AIDanceCoach/issues
2. If not, click: **New Issue**
3. Fill in the template:

```markdown
## Description
Clear description of the bug.

## Steps to Reproduce
1. Do this
2. Then this
3. It happens

## Expected Behavior
What should happen.

## Actual Behavior
What actually happens.

## Environment
- OS: Windows 10 / macOS / Linux
- Python: 3.9.13
- PyTorch: 2.1.2
- GPU: Yes / No

## Error Message
```
Full error traceback here
```

## Screenshots (if applicable)
Add images/videos if helpful
```

---

## 💡 Feature Requests

Have an idea for a new feature?

1. Open an issue with tag: **enhancement**
2. Describe what you want and why
3. Discuss in comments
4. If approved, you can implement it!

Example:
```markdown
## Feature Request: Real-time scoring display

### Problem
Users can't see their score while dancing (only during playback).

### Solution
Display live score in the bottom corner during real-time feedback.

### Additional Context
This would help users understand what they're doing right/wrong immediately.
```

---

## 📝 Style Guide

### **Python Code Style**

```python
# ✅ Good
def calculate_dtw_distance(ref_pose, user_pose):
    """Calculate DTW distance between two poses."""
    distance = compute_dtw(ref_pose, user_pose)
    return distance

# ❌ Bad
def calc_dtw(r, u):
    return compute_dtw(r, u)
```

### **Commit Message Format**

```
[type]: [description]

[optional body with more details]

[optional footer with issue reference]
```

Examples:
```
feature: Add real-time score display to UI
fix: Resolve CUDA memory leak in pose tracker
docs: Update installation guide for Python 3.9
test: Add coverage for DTW scorer edge cases
```

### **Branch Naming**

```
feature/short-description
fix/issue-number-description
docs/what-changed
test/what-tested
refactor/module-improved
```

---

## 🎯 Good First Issues

Looking for where to start? Look for issues tagged:
- `good first issue` - Perfect for beginners
- `help wanted` - Community input needed
- `documentation` - No coding, just writing

---

## 📚 Useful Resources

- [Git Documentation](https://git-scm.com/doc)
- [Python PEP 8](https://www.python.org/dev/peps/pep-0008/)
- [Pytest Documentation](https://docs.pytest.org/)
- [OpenMMLab Contributing](https://github.com/open-mmlab/mmcv/blob/master/CONTRIBUTING.md)

---

## 🆘 Need Help?

- **Installation issues?** → See [FIRST_TIME_SETUP.md](FIRST_TIME_SETUP.md)
- **Stuck on Git?** → Comment on your issue or PR
- **Architecture questions?** → Check [README.md](README.md)
- **Not sure if feature is good?** → Open an issue first!

---

## ✨ What We Love to See

- 🎯 Clear, focused PRs (one feature per PR)
- 📝 Well-documented code
- ✅ Tests for new features
- 💬 Thoughtful commit messages
- 🤝 Friendly, collaborative attitude

---

## 🎉 Thank You!

Every contribution makes this project better. We appreciate:
- Code contributions
- Bug reports
- Feature ideas
- Documentation improvements
- User feedback

**Welcome to the team!** 🚀

---

**Happy contributing! 🎬✨**

Questions? Open an issue or reach out to the maintainers!
