import os
import sys
import subprocess

def check_dependencies():
    """
    Checks environment libraries and auto-installs missing ones.
    Bypasses halts on failure by printing warnings.
    """
    required_packages = {
        'torch': 'torch',
        'monai': 'monai',
        'plotly': 'plotly',
        'pydantic': 'pydantic>=2.0',
        'streamlit': 'streamlit',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'nibabel': 'nibabel',
        'scipy': 'scipy'
    }
    
    print("=" * 60)
    print("PROJECT #45: SWIN 3D CLINICAL COCKPIT BOOTSTRAP")
    print("=" * 60)
    
    for module_name, pip_name in required_packages.items():
        try:
            __import__(module_name)
            print(f"  [OK] {module_name:<12} : Already installed.")
        except ImportError:
            print(f"  [..] {module_name:<12} : Missing. Installing {pip_name}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
                print(f"  [OK] {module_name:<12} : Successfully installed.")
            except Exception as err:
                print(f"  [WARN] Failed to install {pip_name}: {err}. Relying on local fallback blocks.")
    print("-" * 60)


def precompile_swin_weights():
    """
    Pre-compiles standard mock weights for SwinCrossAttentionNet
    to guarantee the Vision-Attention diagnostic branch boots instantly.
    """
    weights_path = os.path.join(os.path.dirname(__file__), "weights_swin.pth")
    if os.path.exists(weights_path):
        print("  [OK] SwinCrossAttentionNet weights found.")
        return
        
    print("  [..] Compiling SwinCrossAttentionNet pre-trained mock checkpoint weights...")
    try:
        import torch
        from swin_fusion_net import SwinCrossAttentionNet
        
        # Instantiate Swin-Fusion Network and save state dictionary
        model = SwinCrossAttentionNet()
        torch.save(model.state_dict(), weights_path)
        print("  [OK] weights_swin.pth successfully created!")
    except Exception as err:
        print(f"  [WARNING] Could not compile weights: {err}")
    print("-" * 60)


def main():
    check_dependencies()
    precompile_swin_weights()
    
    app_path = os.path.join(os.path.dirname(__file__), "app_clinical_3d.py")
    print(f"\n[LAUNCHING] Starting Swin-Net 3D dashboard on {app_path}...\n")
    
    try:
        # Launch Streamlit cockpit programmatically in headless mode
        subprocess.run([sys.executable, "-m", "streamlit", "run", app_path, "--server.headless=true"])
    except KeyboardInterrupt:
        print("\n[STOPPED] Swin-Net dashboard terminated by user.")
    except Exception as err:
        print(f"[FATAL] Streamlit failed to launch: {err}")


if __name__ == "__main__":
    main()
