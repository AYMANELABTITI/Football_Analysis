"""
Module de tracking des joueurs et du ballon
Utilise SORT (Simple Online and Realtime Tracking) avec Kalman filter
"""

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple, Union
import yaml


class KalmanBoxTracker:
    """
    Tracker Kalman pour une boîte de détection
    Prédit la position future basée sur les observations passées
    """
    count = 0
    
    def __init__(self, bbox):
        """
        Initialise un tracker pour une bbox
        bbox: [x1, y1, x2, y2, conf, class]
        """
        # Définir le filtre de Kalman
        # État: [x, y, s, r, vx, vy, vs]
        # x, y: centre de la bbox
        # s: surface (scale)
        # r: ratio aspect
        # vx, vy, vs: vélocités
        
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        
        # Matrice de transition d'état
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        
        # Matrice de mesure
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        
        # Covariance de mesure
        self.kf.R[2:, 2:] *= 10.
        
        # Covariance du processus
        self.kf.P[4:, 4:] *= 1000.
        self.kf.P *= 10.
        
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        
        # Initialiser l'état
        self.kf.x[:4] = self._convert_bbox_to_z(bbox)
        
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        
        # Pour le calcul de distance
        self.positions = []
        self.timestamps = []
    
    def _convert_bbox_to_z(self, bbox):
        """
        Convertit [x1, y1, x2, y2] en [x, y, s, r]
        x, y: centre
        s: surface
        r: ratio (w/h)
        """
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w/2.
        y = bbox[1] + h/2.
        s = w * h
        r = w / float(h) if h != 0 else 0
        return np.array([x, y, s, r]).reshape((4, 1))
    
    def _convert_x_to_bbox(self, x):
        """
        Convertit [x, y, s, r] en [x1, y1, x2, y2]
        """
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w if w != 0 else 0
        return np.array([
            x[0] - w/2.,
            x[1] - h/2.,
            x[0] + w/2.,
            x[1] + h/2.
        ]).reshape((1, 4))
    
    def update(self, bbox, timestamp=None):
        """
        Met à jour le tracker avec une nouvelle détection
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self._convert_bbox_to_z(bbox))
        
        # Enregistrer la position pour le calcul de distance
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        self.positions.append((center_x, center_y))
        if timestamp is not None:
            self.timestamps.append(timestamp)
    
    def predict(self):
        """
        Prédit la position future
        """
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] *= 0.0
        
        self.kf.predict()
        self.age += 1
        
        if self.time_since_update > 0:
            self.hit_streak = 0
        
        self.time_since_update += 1
        self.history.append(self._convert_x_to_bbox(self.kf.x))
        
        return self.history[-1]
    
    def get_state(self):
        """
        Retourne l'état actuel de la bbox
        """
        return self._convert_x_to_bbox(self.kf.x)
    
    def get_distance_traveled(self, fps=30):
        """
        Calcule la distance parcourue en pixels (peut être convertie en mètres)
        """
        if len(self.positions) < 2:
            return 0.0
        
        distance = 0.0
        for i in range(1, len(self.positions)):
            x1, y1 = self.positions[i-1]
            x2, y2 = self.positions[i]
            distance += np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        return distance


class FootballTracker:
    """
    Tracker principal pour suivre plusieurs joueurs et le ballon
    """
    
    def __init__(self, config_path: Union[str, Dict] = "config/config.yaml"):
        """
        Initialise le tracker
        """
        if isinstance(config_path, dict):
            self.config = config_path
        else:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        
        self.max_age = self.config['tracking']['max_age']
        self.min_hits = self.config['tracking']['min_hits']
        self.iou_threshold = self.config['tracking']['iou_threshold']
        
        self.trackers = []
        self.frame_count = 0
        
        print(f"✅ Tracker initialisé (max_age={self.max_age}, min_hits={self.min_hits})")
    
    def _iou(self, bb_test, bb_gt):
        """
        Calcule l'Intersection over Union (IoU) entre deux bboxes
        """
        xx1 = np.maximum(bb_test[0], bb_gt[0])
        yy1 = np.maximum(bb_test[1], bb_gt[1])
        xx2 = np.minimum(bb_test[2], bb_gt[2])
        yy2 = np.minimum(bb_test[3], bb_gt[3])
        
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        
        intersection = w * h
        area_test = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
        area_gt = (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
        union = area_test + area_gt - intersection
        
        iou = intersection / union if union > 0 else 0
        return iou
    
    def update(self, detections: List) -> List[Dict]:
        """
        Met à jour les trackers avec les nouvelles détections
        
        Args:
            detections: Liste de détections [[x1, y1, x2, y2, conf, class], ...]
            
        Returns:
            Liste de tracks actifs avec leurs IDs
        """
        self.frame_count += 1
        
        # Obtenir les prédictions des trackers existants
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        
        for t in reversed(to_del):
            self.trackers.pop(t)
        
        # Association détections-trackers
        if len(detections) > 0:
            detections_array = np.array(detections)
            matched, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(
                detections_array, trks
            )
            
            # Mettre à jour les trackers matchés
            for m in matched:
                self.trackers[m[1]].update(detections_array[m[0], :], self.frame_count)
            
            # Créer de nouveaux trackers pour les détections non matchées
            for i in unmatched_dets:
                trk = KalmanBoxTracker(detections_array[i, :])
                self.trackers.append(trk)
        
        # Retourner les tracks actifs
        ret = []
        for trk in self.trackers:
            d = trk.get_state()[0]
            
            if (trk.time_since_update < 1) and \
               (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append({
                    'id': trk.id,
                    'bbox': d[:4],
                    'hits': trk.hits,
                    'age': trk.age,
                    'positions': trk.positions,
                    'distance': trk.get_distance_traveled()
                })
        
        # Supprimer les trackers morts
        self.trackers = [t for t in self.trackers if t.time_since_update < self.max_age]
        
        return ret
    
    def _associate_detections_to_trackers(self, detections, trackers):
        """
        Associe les détections aux trackers en utilisant l'IoU
        """
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
        
        iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
        
        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_matrix[d, t] = self._iou(det[:4], trk[:4])
        
        # Résoudre l'assignation avec l'algorithme hongrois
        if min(iou_matrix.shape) > 0:
            matched_indices = linear_sum_assignment(-iou_matrix)
            matched_indices = np.array(list(zip(*matched_indices)))
        else:
            matched_indices = np.empty(shape=(0, 2))
        
        unmatched_detections = []
        for d in range(len(detections)):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)
        
        unmatched_trackers = []
        for t in range(len(trackers)):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)
        
        # Filtrer les matchs avec IoU faible
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
        
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)
        
        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
    
    def get_statistics(self) -> Dict:
        """
        Obtient les statistiques de tracking
        """
        stats = {
            'total_tracks': len(self.trackers),
            'active_tracks': len([t for t in self.trackers if t.time_since_update < 1]),
            'frame_count': self.frame_count
        }
        return stats


if __name__ == "__main__":
    # Test du tracker
    tracker = FootballTracker()
    print("✅ Tracker initialisé avec succès")
