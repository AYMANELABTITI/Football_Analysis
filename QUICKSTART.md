# 🚀 Guide de Démarrage Rapide

## Installation en 5 Minutes

### 1. Vérifier Python
```bash
python --version  # Doit être 3.8 ou supérieur
```

### 2. Installer les dépendances
```bash
# Avec GPU (recommandé)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Sans GPU (plus lent)
pip install -r requirements.txt
```

### 3. Tester l'installation
```bash
python test_system.py
```

### 4. Analyser votre première vidéo
```bash
# Placer votre vidéo dans data/raw/
python main.py data/raw/match.mp4
```

## ⚡ Commandes Essentielles

### Analyse Simple
```bash
python main.py data/raw/match.mp4
```

### Avec Prévisualisation
```bash
python main.py data/raw/match.mp4 --preview
```

### Personnaliser la Sortie
```bash
python main.py data/raw/match.mp4 --output mon_analyse.mp4
```

## 🎯 Que Fait le Système ?

1. **Détecte** les joueurs et le ballon
2. **Suit** leurs mouvements (tracking)
3. **Identifie** les événements (hors-jeu, corners, etc.)
4. **Génère**:
   - Vidéo annotée avec trajectoires
   - Statistiques CSV par joueur
   - Rapport visuel avec graphiques
   - Résumé textuel

## 📁 Où Trouver les Résultats ?

```
results/
├── videos/
│   └── analyzed_match.mp4        # Vidéo annotée
├── stats/
│   ├── match_players.csv         # Stats joueurs
│   ├── match_events.csv          # Événements
│   ├── match_report.png          # Graphiques
│   └── match_summary.txt         # Résumé
```

## ⚙️ Configuration Rapide

Éditez `config/config.yaml`:

```yaml
# Activer/désactiver GPU
gpu:
  enabled: true        # false pour CPU

# Qualité de détection
detection:
  model: "yolov8x.pt"  # yolov8n.pt pour plus rapide
  confidence: 0.3      # 0.5 pour plus précis

# Persistence du tracking
tracking:
  max_age: 30          # Combien de temps garder un track
  min_hits: 3          # Minimum de détections pour valider
```

## 🐛 Problèmes Courants

### GPU non détecté
```bash
# Vérifier CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Si False, installer PyTorch avec CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Mémoire insuffisante
```yaml
# Dans config.yaml
detection:
  model: "yolov8n.pt"  # Modèle plus léger
  img_size: 640        # Réduire la résolution
```

### Détections manquées
```yaml
# Dans config.yaml
detection:
  confidence: 0.2      # Seuil plus bas
tracking:
  min_hits: 2          # Moins strict
```

## 📊 Exemple de Résultats

Après analyse, vous obtenez:

**Statistiques Joueurs:**
| Player | Distance (m) | Vitesse Moy (km/h) |
|--------|-------------|-------------------|
| P_001  | 8,234       | 12.4              |
| P_002  | 7,891       | 11.8              |

**Événements:**
- Hors-jeu: 8
- Corners: 12
- Passes: 245
- Tirs: 18

## 💡 Astuces

### Optimiser la Performance
- Utiliser YOLOv8n pour traitement rapide
- Réduire `img_size` si mémoire limitée
- Désactiver `--preview` pour traitement batch

### Améliorer la Précision
- Utiliser YOLOv8x pour meilleure détection
- Augmenter `confidence` pour moins de faux positifs
- Calibrer les couleurs d'équipe si mal classifiées

### Traiter Plusieurs Vidéos
```bash
for video in data/raw/*.mp4; do
    python main.py "$video"
done
```

## 📞 Support

Consultez:
- `README.md` - Documentation complète
- `test_system.py` - Tests diagnostiques
- Modules individuels pour debug spécifique

## 🎓 Prochaines Étapes

1. ✅ Analyser votre première vidéo
2. 📊 Explorer les statistiques générées
3. ⚙️ Personnaliser la configuration
4. 🎨 Modifier les couleurs de visualisation
5. 🔧 Ajouter vos propres événements

---

**Bon match! ⚽🏟️**
