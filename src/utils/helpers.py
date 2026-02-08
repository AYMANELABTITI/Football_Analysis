"""
Fonctions utilitaires pour l'analyse de football
"""

import cv2
import numpy as np
from typing import Tuple, List
import torch


def check_gpu_availability():
    """
    Vérifie la disponibilité du GPU et affiche les informations
    """
    print("\n" + "="*60)
    print("🔍 VÉRIFICATION DU GPU")
    print("="*60)
    
    if torch.cuda.is_available():
        print(f"✅ GPU disponible!")
        print(f"   Nom: {torch.cuda.get_device_name(0)}")
        print(f"   Nombre de GPUs: {torch.cuda.device_count()}")
        print(f"   CUDA Version: {torch.version.cuda}")
        
        # Mémoire
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        allocated = torch.cuda.memory_allocated(0) / 1e9
        cached = torch.cuda.memory_reserved(0) / 1e9
        
        print(f"   Mémoire totale: {total_memory:.2f} GB")
        print(f"   Mémoire allouée: {allocated:.2f} GB")
        print(f"   Mémoire en cache: {cached:.2f} GB")
        print(f"   Mémoire libre: {total_memory - allocated:.2f} GB")
        
        return True
    else:
        print("❌ GPU non disponible")
        print("   Le traitement utilisera le CPU (plus lent)")
        return False


def estimate_homography(frame: np.ndarray, field_points: List[Tuple] = None) -> np.ndarray:
    """
    Estime la matrice d'homographie pour transformer l'image vers vue terrain
    
    Args:
        frame: Image du terrain
        field_points: Points de référence sur le terrain (optionnel)
        
    Returns:
        Matrice d'homographie 3x3
    """
    if field_points is None:
        # Points par défaut (à ajuster manuellement pour chaque vidéo)
        # Format: (x_pixel, y_pixel) -> (x_terrain, y_terrain)
        src_points = np.float32([
            [100, 100],   # Coin haut-gauche
            [1820, 100],  # Coin haut-droit
            [100, 980],   # Coin bas-gauche
            [1820, 980]   # Coin bas-droit
        ])
        
        # Points correspondants sur un terrain 105x68m
        dst_points = np.float32([
            [0, 0],
            [105, 0],
            [0, 68],
            [105, 68]
        ])
    else:
        src_points = np.float32([p[0] for p in field_points])
        dst_points = np.float32([p[1] for p in field_points])
    
    # Calculer l'homographie
    H, status = cv2.findHomography(src_points, dst_points)
    
    return H


def pixel_distance_to_meters(pixel_distance: float, homography: np.ndarray,
                             reference_point: Tuple = (960, 540)) -> float:
    """
    Convertit une distance en pixels en mètres
    
    Args:
        pixel_distance: Distance en pixels
        homography: Matrice d'homographie
        reference_point: Point de référence pour le calcul
        
    Returns:
        Distance en mètres
    """
    # Point de référence
    p1 = np.array([reference_point[0], reference_point[1], 1.0])
    
    # Point décalé
    p2 = np.array([reference_point[0] + pixel_distance, reference_point[1], 1.0])
    
    # Transformer en coordonnées terrain
    p1_field = homography @ p1
    p2_field = homography @ p2
    
    p1_field = p1_field[:2] / p1_field[2]
    p2_field = p2_field[:2] / p2_field[2]
    
    # Calculer la distance euclidienne
    distance_m = np.linalg.norm(p2_field - p1_field)
    
    return distance_m


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calcule l'Intersection over Union entre deux boîtes
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
        
    Returns:
        IoU score (0-1)
    """
    # Coordonnées de l'intersection
    x1_i = max(box1[0], box2[0])
    y1_i = max(box1[1], box2[1])
    x2_i = min(box1[2], box2[2])
    y2_i = min(box1[3], box2[3])
    
    # Surface d'intersection
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Surfaces des boîtes
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Union
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


def non_max_suppression(boxes: List, iou_threshold: float = 0.5) -> List:
    """
    Applique la suppression non-maximale pour éliminer les détections redondantes
    
    Args:
        boxes: Liste de boîtes [x1, y1, x2, y2, confidence]
        iou_threshold: Seuil IoU
        
    Returns:
        Boîtes filtrées
    """
    if len(boxes) == 0:
        return []
    
    # Convertir en array numpy
    boxes_array = np.array(boxes)
    
    # Extraire les coordonnées et confidences
    x1 = boxes_array[:, 0]
    y1 = boxes_array[:, 1]
    x2 = boxes_array[:, 2]
    y2 = boxes_array[:, 3]
    scores = boxes_array[:, 4]
    
    # Calculer les surfaces
    areas = (x2 - x1) * (y2 - y1)
    
    # Trier par score décroissant
    order = scores.argsort()[::-1]
    
    keep = []
    
    while len(order) > 0:
        # Prendre la boîte avec le score le plus élevé
        i = order[0]
        keep.append(i)
        
        # Calculer l'IoU avec les autres boîtes
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        
        intersection = w * h
        iou = intersection / (areas[i] + areas[order[1:]] - intersection)
        
        # Garder les boîtes avec IoU < seuil
        indices = np.where(iou <= iou_threshold)[0]
        order = order[indices + 1]
    
    return boxes_array[keep].tolist()


def smooth_trajectory(positions: List[Tuple], window_size: int = 5) -> List[Tuple]:
    """
    Lisse une trajectoire avec un filtre moyenne mobile
    
    Args:
        positions: Liste de positions (x, y)
        window_size: Taille de la fenêtre de lissage
        
    Returns:
        Positions lissées
    """
    if len(positions) < window_size:
        return positions
    
    smoothed = []
    
    for i in range(len(positions)):
        start = max(0, i - window_size // 2)
        end = min(len(positions), i + window_size // 2 + 1)
        
        window = positions[start:end]
        avg_x = np.mean([p[0] for p in window])
        avg_y = np.mean([p[1] for p in window])
        
        smoothed.append((avg_x, avg_y))
    
    return smoothed


def extract_dominant_color(image: np.ndarray, k: int = 3) -> Tuple[int, int, int]:
    """
    Extrait la couleur dominante d'une image avec k-means
    
    Args:
        image: Image BGR
        k: Nombre de clusters
        
    Returns:
        Couleur dominante (B, G, R)
    """
    # Reshape l'image
    pixels = image.reshape(-1, 3).astype(np.float32)
    
    # Appliquer k-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10,
                                    cv2.KMEANS_RANDOM_CENTERS)
    
    # Trouver le cluster le plus fréquent
    unique, counts = np.unique(labels, return_counts=True)
    dominant_idx = unique[np.argmax(counts)]
    
    dominant_color = centers[dominant_idx].astype(int)
    
    return tuple(dominant_color.tolist())


def draw_field_lines(frame: np.ndarray, homography: np.ndarray = None) -> np.ndarray:
    """
    Dessine les lignes du terrain sur la frame
    
    Args:
        frame: Image
        homography: Matrice d'homographie (optionnel)
        
    Returns:
        Frame avec lignes
    """
    overlay = frame.copy()
    
    # Si pas d'homographie, dessiner des lignes simples
    h, w = frame.shape[:2]
    
    # Ligne médiane
    cv2.line(overlay, (w // 2, 0), (w // 2, h), (255, 255, 255), 2)
    
    # Cercle central
    cv2.circle(overlay, (w // 2, h // 2), 80, (255, 255, 255), 2)
    
    # Surfaces de réparation (approximatives)
    penalty_w = w // 6
    penalty_h = h // 3
    
    # Gauche
    cv2.rectangle(overlay, (0, h // 2 - penalty_h // 2),
                 (penalty_w, h // 2 + penalty_h // 2), (255, 255, 255), 2)
    
    # Droite
    cv2.rectangle(overlay, (w - penalty_w, h // 2 - penalty_h // 2),
                 (w, h // 2 + penalty_h // 2), (255, 255, 255), 2)
    
    # Blend avec l'image originale
    alpha = 0.3
    result = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    return result


def create_video_from_frames(frames: List[np.ndarray], output_path: str,
                            fps: int = 30):
    """
    Crée une vidéo à partir d'une liste de frames
    
    Args:
        frames: Liste de frames
        output_path: Chemin de sortie
        fps: Frames par seconde
    """
    if len(frames) == 0:
        print("❌ Aucune frame à sauvegarder")
        return
    
    height, width = frames[0].shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(frame)
    
    out.release()
    
    print(f"✅ Vidéo sauvegardée: {output_path}")


def benchmark_gpu():
    """
    Effectue un benchmark simple du GPU
    """
    print("\n🏃 Benchmark GPU...")
    
    if not torch.cuda.is_available():
        print("❌ GPU non disponible pour le benchmark")
        return
    
    device = torch.device("cuda:0")
    
    # Test de calcul matriciel
    size = 5000
    print(f"   Test: Multiplication de matrices {size}x{size}")
    
    import time
    
    # GPU
    start = time.time()
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    
    # CPU
    start = time.time()
    a_cpu = torch.randn(size, size)
    b_cpu = torch.randn(size, size)
    c_cpu = torch.matmul(a_cpu, b_cpu)
    cpu_time = time.time() - start
    
    print(f"   Temps GPU: {gpu_time:.4f}s")
    print(f"   Temps CPU: {cpu_time:.4f}s")
    print(f"   Accélération: {cpu_time/gpu_time:.2f}x")


if __name__ == "__main__":
    # Tests
    check_gpu_availability()
    
    # Test IoU
    box1 = [0, 0, 100, 100]
    box2 = [50, 50, 150, 150]
    iou = calculate_iou(box1, box2)
    print(f"\n📐 Test IoU: {iou:.3f}")
    
    # Benchmark
    if torch.cuda.is_available():
        benchmark_gpu()
