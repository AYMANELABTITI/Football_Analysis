@echo off
REM Script d'installation pour Windows
REM Usage: install.bat

echo ======================================================================
echo INSTALLATION DU SYSTEME D'ANALYSE DE FOOTBALL
echo ======================================================================
echo.

REM 1. Vérifier Python
echo Verification de Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERREUR] Python n'est pas installe ou n'est pas dans le PATH
    pause
    exit /b 1
)
python --version
echo [OK] Python trouve
echo.

REM 2. Créer l'environnement virtuel
echo Creation de l'environnement virtuel...
if not exist venv (
    python -m venv venv
    echo [OK] Environnement virtuel cree
) else (
    echo [INFO] Environnement virtuel existe deja
)
echo.

REM 3. Activer l'environnement virtuel
echo Activation de l'environnement virtuel...
call venv\Scripts\activate.bat
echo [OK] Environnement virtuel active
echo.

REM 4. Mettre à jour pip
echo Mise a jour de pip...
python -m pip install --upgrade pip >nul 2>&1
echo [OK] pip mis a jour
echo.

REM 5. Installer PyTorch avec CUDA
echo Installation de PyTorch...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [INFO] Aucun GPU NVIDIA detecte
    echo [INFO] Installation de PyTorch CPU uniquement...
    pip install torch torchvision torchaudio
) else (
    echo [OK] GPU NVIDIA detecte
    echo [INFO] Installation de PyTorch avec support CUDA...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
)
echo [OK] PyTorch installe
echo.

REM 6. Installer les dépendances
echo Installation des dependances...
pip install -r requirements.txt
echo [OK] Dependances installees
echo.

REM 7. Créer les dossiers
echo Creation de la structure de dossiers...
type nul > data\raw\.gitkeep
type nul > data\processed\.gitkeep
type nul > models\.gitkeep
type nul > results\videos\.gitkeep
type nul > results\stats\.gitkeep
type nul > results\reports\.gitkeep
echo [OK] Structure de dossiers creee
echo.

REM 8. Télécharger le modèle
echo Telechargement du modele YOLOv8...
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" >nul 2>&1
echo [OK] Modele YOLOv8n telecharge
echo.

REM 9. Test
echo Test de l'installation...
python test_system.py
echo.

REM 10. Résumé
echo ======================================================================
echo INSTALLATION TERMINEE!
echo ======================================================================
echo.
echo Prochaines etapes:
echo.
echo 1. Activer l'environnement virtuel:
echo    venv\Scripts\activate.bat
echo.
echo 2. Analyser une video:
echo    python main.py data\raw\votre_match.mp4
echo.
echo 3. Ou lancer la demonstration:
echo    python demo.py --interactive
echo.
echo Documentation:
echo    - README.md (documentation complete)
echo    - QUICKSTART.md (guide de demarrage rapide)
echo.
echo ======================================================================
pause
