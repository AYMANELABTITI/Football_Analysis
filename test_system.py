"""
Script de test pour vérifier l'installation et les composants
"""

import sys
import os

# Ajouter le chemin
sys.path.append(os.path.dirname(__file__))

def test_imports():
    """Test des imports de base"""
    print("="*60)
    print("🧪 TEST DES IMPORTS")
    print("="*60)
    
    tests = {
        'numpy': 'NumPy',
        'cv2': 'OpenCV',
        'torch': 'PyTorch',
        'ultralytics': 'YOLOv8',
        'filterpy': 'FilterPy',
        'matplotlib': 'Matplotlib',
        'pandas': 'Pandas',
        'yaml': 'PyYAML'
    }
    
    results = {}
    
    for module, name in tests.items():
        try:
            __import__(module)
            results[name] = '✅ OK'
        except ImportError as e:
            results[name] = f'❌ ERREUR: {e}'
    
    for name, status in results.items():
        print(f"{name:15} {status}")
    
    return all('✅' in r for r in results.values())


def test_gpu():
    """Test de la disponibilité GPU"""
    print("\n" + "="*60)
    print("🔍 TEST GPU")
    print("="*60)
    
    import torch
    
    if torch.cuda.is_available():
        print(f"✅ GPU disponible")
        print(f"   Device: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA: {torch.version.cuda}")
        print(f"   Mémoire: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        return True
    else:
        print("⚠️  GPU non disponible - utilisation du CPU")
        return False


def test_yolo():
    """Test du modèle YOLO"""
    print("\n" + "="*60)
    print("🤖 TEST YOLO")
    print("="*60)
    
    try:
        from ultralytics import YOLO
        import numpy as np
        
        # Charger le modèle
        print("Téléchargement du modèle YOLOv8n (peut prendre quelques minutes)...")
        model = YOLO('yolov8n.pt')
        
        # Test sur image factice
        test_img = np.zeros((640, 640, 3), dtype=np.uint8)
        results = model(test_img, verbose=False)
        
        print("✅ YOLOv8 fonctionne correctement")
        return True
        
    except Exception as e:
        print(f"❌ Erreur YOLO: {e}")
        return False


def test_modules():
    """Test des modules du projet"""
    print("\n" + "="*60)
    print("📦 TEST DES MODULES")
    print("="*60)
    
    modules = {
        'Detector': 'src.detection.detector',
        'Tracker': 'src.tracking.tracker',
        'EventDetector': 'src.events.event_detector',
        'Visualizer': 'src.visualization.visualizer',
        'Helpers': 'src.utils.helpers'
    }
    
    results = {}
    
    for name, module_path in modules.items():
        try:
            __import__(module_path)
            results[name] = '✅ OK'
        except Exception as e:
            results[name] = f'❌ ERREUR: {e}'
    
    for name, status in results.items():
        print(f"{name:15} {status}")
    
    return all('✅' in r for r in results.values())


def test_config():
    """Test du fichier de configuration"""
    print("\n" + "="*60)
    print("⚙️  TEST CONFIGURATION")
    print("="*60)
    
    try:
        import yaml
        
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Vérifier les sections principales
        required_sections = ['gpu', 'detection', 'tracking', 'field', 'events', 'video']
        
        for section in required_sections:
            if section in config:
                print(f"✅ Section '{section}' présente")
            else:
                print(f"❌ Section '{section}' manquante")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur configuration: {e}")
        return False


def test_directories():
    """Test de la structure des dossiers"""
    print("\n" + "="*60)
    print("📁 TEST STRUCTURE")
    print("="*60)
    
    required_dirs = [
        'data/raw',
        'data/processed',
        'models',
        'results/videos',
        'results/stats',
        'results/reports',
        'config',
        'src/detection',
        'src/tracking',
        'src/events',
        'src/visualization',
        'src/utils'
    ]
    
    all_ok = True
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path} manquant")
            all_ok = False
    
    return all_ok


def run_all_tests():
    """Exécute tous les tests"""
    print("\n")
    print("🏟️  TEST COMPLET DU SYSTÈME D'ANALYSE DE FOOTBALL")
    print("="*60)
    
    results = {
        'Imports': test_imports(),
        'GPU': test_gpu(),
        'YOLO': test_yolo(),
        'Modules': test_modules(),
        'Configuration': test_config(),
        'Structure': test_directories()
    }
    
    print("\n" + "="*60)
    print("📊 RÉSUMÉ DES TESTS")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASSÉ" if passed else "❌ ÉCHOUÉ"
        print(f"{test_name:20} {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ TOUS LES TESTS SONT PASSÉS!")
        print("🚀 Le système est prêt à être utilisé")
    else:
        print("❌ CERTAINS TESTS ONT ÉCHOUÉ")
        print("⚠️  Veuillez corriger les erreurs avant utilisation")
    print("="*60)
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
