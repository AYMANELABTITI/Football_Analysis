#!/bin/bash

# Script d'installation automatique pour le système d'analyse de football
# Usage: bash install.sh

set -e  # Arrêter en cas d'erreur

echo "======================================================================"
echo "🏟️  INSTALLATION DU SYSTÈME D'ANALYSE DE FOOTBALL"
echo "======================================================================"

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Fonction pour afficher les messages
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${NC}ℹ️  $1${NC}"
}

# 1. Vérifier Python
echo ""
print_info "Vérification de Python..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    print_success "Python $PYTHON_VERSION trouvé"
else
    print_error "Python 3 n'est pas installé"
    exit 1
fi

# 2. Créer l'environnement virtuel
echo ""
print_info "Création de l'environnement virtuel..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_success "Environnement virtuel créé"
else
    print_warning "Environnement virtuel existe déjà"
fi

# 3. Activer l'environnement virtuel
echo ""
print_info "Activation de l'environnement virtuel..."
source venv/bin/activate
print_success "Environnement virtuel activé"

# 4. Mettre à jour pip
echo ""
print_info "Mise à jour de pip..."
pip install --upgrade pip > /dev/null 2>&1
print_success "pip mis à jour"

# 5. Installer PyTorch avec CUDA (si disponible)
echo ""
print_info "Installation de PyTorch..."

# Détecter si CUDA est disponible
if command -v nvidia-smi &> /dev/null; then
    print_success "GPU NVIDIA détecté"
    print_info "Installation de PyTorch avec support CUDA..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    print_success "PyTorch avec CUDA installé"
else
    print_warning "Aucun GPU NVIDIA détecté"
    print_info "Installation de PyTorch CPU uniquement..."
    pip install torch torchvision torchaudio
    print_success "PyTorch CPU installé"
fi

# 6. Installer les autres dépendances
echo ""
print_info "Installation des dépendances..."
pip install -r requirements.txt
print_success "Dépendances installées"

# 7. Créer les dossiers .gitkeep
echo ""
print_info "Création de la structure de dossiers..."
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch models/.gitkeep
touch results/videos/.gitkeep
touch results/stats/.gitkeep
touch results/reports/.gitkeep
print_success "Structure de dossiers créée"

# 8. Télécharger le modèle YOLOv8
echo ""
print_info "Téléchargement du modèle YOLOv8..."
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" > /dev/null 2>&1
print_success "Modèle YOLOv8n téléchargé"

# 9. Tester l'installation
echo ""
print_info "Test de l'installation..."
python test_system.py

# 10. Résumé
echo ""
echo "======================================================================"
echo "✅ INSTALLATION TERMINÉE!"
echo "======================================================================"
echo ""
echo "📋 Prochaines étapes:"
echo ""
echo "1. Activer l'environnement virtuel:"
echo "   source venv/bin/activate"
echo ""
echo "2. Analyser une vidéo:"
echo "   python main.py data/raw/votre_match.mp4"
echo ""
echo "3. Ou lancer la démonstration:"
echo "   python demo.py --interactive"
echo ""
echo "📚 Documentation:"
echo "   - README.md (documentation complète)"
echo "   - QUICKSTART.md (guide de démarrage rapide)"
echo ""
echo "======================================================================"
