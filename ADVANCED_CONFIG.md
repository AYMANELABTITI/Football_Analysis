# 🔧 Guide de Configuration Avancée

## Configuration GPU

### Optimisation Mémoire

Pour vidéos 4K ou GPU avec mémoire limitée:

```yaml
gpu:
  enabled: true
  device: "cuda:0"
  mixed_precision: true  # Utilise FP16 pour économiser 50% de mémoire

detection:
  img_size: 640         # Réduire de 1280 → 640
  model: "yolov8n.pt"   # Modèle nano (plus léger)
```

### Multi-GPU

Pour utiliser plusieurs GPUs:

```python
# Dans main.py, modifier:
self.detector = FootballDetector(config_path)

# En:
self.detector = FootballDetector(config_path)
torch.cuda.set_device(1)  # Utiliser GPU 1
```

### Batch Processing

Ajuster la taille du batch selon la mémoire GPU:

```python
# Dans detector.py, méthode detect_video()
batch_size = 8  # Par défaut

# GPU 24GB: batch_size = 16
# GPU 12GB: batch_size = 8
# GPU 6GB:  batch_size = 4
```

## Configuration de Détection

### Seuils de Confiance

```yaml
detection:
  confidence: 0.3  # Valeur par défaut
  
# Pour moins de faux positifs:
  confidence: 0.5
  
# Pour plus de détections (avec bruit):
  confidence: 0.2
```

### Sélection du Modèle

| Modèle | Vitesse | Précision | Mémoire |
|--------|---------|-----------|---------|
| yolov8n.pt | +++++ | +++ | 3 GB |
| yolov8s.pt | ++++ | ++++ | 6 GB |
| yolov8m.pt | +++ | +++++ | 9 GB |
| yolov8l.pt | ++ | ++++++ | 12 GB |
| yolov8x.pt | + | +++++++ | 16 GB |

```yaml
detection:
  model: "yolov8x.pt"  # Maximum précision
  model: "yolov8n.pt"  # Maximum vitesse
```

## Configuration de Tracking

### SORT Parameters

```yaml
tracking:
  max_age: 30       # Frames à garder un track sans détection
  min_hits: 3       # Détections minimum pour valider un track
  iou_threshold: 0.3  # Seuil IoU pour matching
```

**Ajustements courants:**

Pour terrain très occupé:
```yaml
tracking:
  max_age: 50      # Plus tolérant
  min_hits: 2      # Moins strict
  iou_threshold: 0.2
```

Pour haute précision:
```yaml
tracking:
  max_age: 20      # Plus strict
  min_hits: 5      # Plus exigeant
  iou_threshold: 0.4
```

## Classification d'Équipes

### Ajuster les Couleurs HSV

Modifier `src/detection/detector.py`:

```python
def classify_teams(self, frame, players):
    # Couleurs HSV
    # H: 0-180 (teinte)
    # S: 0-255 (saturation)
    # V: 0-255 (valeur/luminosité)
    
    # Équipe 1: Rouge
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    # Équipe 2: Bleu
    lower_blue = np.array([100, 100, 100])
    upper_blue = np.array([130, 255, 255])
```

### K-Means pour Classification Automatique

```python
from sklearn.cluster import KMeans

def classify_with_kmeans(self, frame, players, n_teams=2):
    colors = []
    for player in players:
        # Extraire couleur dominante
        roi = frame[y1:y2, x1:x2]
        dominant = self.extract_dominant_color(roi)
        colors.append(dominant)
    
    # Clustering
    kmeans = KMeans(n_clusters=n_teams)
    labels = kmeans.fit_predict(colors)
    
    return labels
```

## Détection d'Événements

### Seuils Personnalisés

```yaml
events:
  offside_threshold: 2.0      # Distance minimum pour hors-jeu (m)
  ball_possession_distance: 1.5  # Distance max pour possession (m)
  high_speed_threshold: 20    # Vitesse pour course (km/h)
  sprint_threshold: 25        # Vitesse pour sprint (km/h)
```

### Calibration du Terrain

Pour transformer pixels → mètres réels:

```python
# Dans main.py, après initialisation
from src.utils.helpers import estimate_homography

# Points du terrain (pixels) → Coordonnées réelles (mètres)
field_points = [
    ((100, 50), (0, 0)),           # Coin haut-gauche
    ((1820, 50), (105, 0)),        # Coin haut-droit
    ((100, 1030), (0, 68)),        # Coin bas-gauche
    ((1820, 1030), (105, 68))      # Coin bas-droit
]

H = estimate_homography(frame, field_points)
event_detector.set_field_homography(H)
```

## Visualisation

### Couleurs Personnalisées

```python
# Dans src/visualization/visualizer.py
self.colors = {
    'team_1': (0, 0, 255),      # Votre couleur BGR
    'team_2': (255, 0, 0),      
    'referee': (50, 50, 50),    
    'ball': (0, 255, 255),      
    'trajectory': (255, 100, 0)
}
```

### Longueur des Trajectoires

```yaml
visualization:
  trajectory_length: 30  # Nombre de points à afficher
```

```python
# Dans main.py
frame = self.visualizer.draw_trajectories(frame, max_length=50)
```

## Performance

### Optimisations

1. **Résolution d'entrée:**
```yaml
video:
  resize_width: 1280   # Au lieu de 1920
  resize_height: 720   # Au lieu de 1080
```

2. **Skip frames:**
```python
# Analyser 1 frame sur 2
if frame_idx % 2 == 0:
    continue
```

3. **ROI Processing:**
```python
# Ne traiter que la partie centrale
roi = frame[100:900, 200:1720]
```

### Profiling

Mesurer les temps:

```python
import time

start = time.time()
players, balls = self.detector.detect_frame(frame)
detection_time = time.time() - start

start = time.time()
tracks = self.tracker.update(players)
tracking_time = time.time() - start

print(f"Detection: {detection_time*1000:.2f}ms")
print(f"Tracking: {tracking_time*1000:.2f}ms")
```

## Export

### Formats Vidéo

```yaml
output:
  video_codec: "mp4v"   # Par défaut
  # Autres options:
  # "avc1" - H.264 (meilleure compatibilité)
  # "XVID" - Xvid
  # "MJPG" - Motion JPEG
```

### Qualité Vidéo

```python
# Dans main.py
fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264
out = cv2.VideoWriter(
    output_path, 
    fourcc, 
    fps, 
    (width, height),
    isColor=True
)

# Pour meilleure qualité
out.set(cv2.VIDEOWRITER_PROP_QUALITY, 100)
```

## Exemple de Configuration Complète

```yaml
# config/production.yaml

gpu:
  enabled: true
  device: "cuda:0"
  mixed_precision: true

detection:
  model: "yolov8x.pt"
  confidence: 0.4
  iou_threshold: 0.5
  img_size: 1280

tracking:
  max_age: 40
  min_hits: 3
  iou_threshold: 0.3

field:
  length: 105
  width: 68
  penalty_area_length: 16.5
  penalty_area_width: 40.3

events:
  offside_threshold: 2.5
  ball_possession_distance: 2.0
  high_speed_threshold: 20
  sprint_threshold: 28

video:
  fps: 30
  output_fps: 30
  resize_width: 1920
  resize_height: 1080

output:
  save_annotated_video: true
  save_trajectories: true
  save_heatmaps: true
  save_statistics: true
  video_codec: "avc1"
```

## Utilisation

```bash
python main.py match.mp4 --config config/production.yaml
```
