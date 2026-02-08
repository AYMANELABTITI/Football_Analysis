"""
Module de détection professionnel avec pose estimation
Version FIFA-like avec skeleton tracking
"""

import torch
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict, Optional
import yaml


class ProfessionalFootballDetector:
    """Détecteur professionnel avec pose estimation et filtrage avancé"""
    
    def __init__(self, config_path: str = "config/config_pro.yaml"):
        """Initialise le détecteur professionnel"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        detection_cfg = self.config.get('detection', {})
        
        # Configuration GPU
        self.device = self._setup_gpu()
        print(f"🖥️  Device: {self.device}")
        
        # Charger YOLO pour détection
        detection_model = detection_cfg['model']
        self.detector = YOLO(detection_model)
        self.detector.to(self.device)
        
        # Charger YOLO-Pose pour skeleton tracking
        print("📦 Chargement du modèle de pose...")
        pose_model = detection_cfg.get('pose_model', 'yolov8x-pose.pt')
        self.pose_model = YOLO(pose_model)
        self.pose_model.to(self.device)
        
        # Paramètres
        self.conf_threshold = detection_cfg['confidence']
        self.iou_threshold = detection_cfg['iou_threshold']
        self.img_size = detection_cfg['img_size']
        self.player_conf_threshold = detection_cfg.get(
            'player_confidence', max(0.25, self.conf_threshold * 0.75)
        )
        self.ball_conf_threshold = detection_cfg.get('ball_confidence', self.conf_threshold)
        self.pose_conf_threshold = detection_cfg.get('pose_confidence', self.player_conf_threshold)
        
        # Filtres de qualité
        self.min_player_area = detection_cfg.get('min_player_area', 800)
        self.max_player_area = detection_cfg.get('max_player_area', 15000)
        self.min_aspect_ratio = detection_cfg.get('min_aspect_ratio', 1.5)
        self.max_aspect_ratio = detection_cfg.get('max_aspect_ratio', 4.0)
        self.field_y_min = detection_cfg.get('field_y_min', 0.1)
        self.field_y_max = detection_cfg.get('field_y_max', 0.9)
        self.ball_field_y_min = detection_cfg.get('ball_field_y_min', 0.2)
        self.ball_field_y_max = detection_cfg.get('ball_field_y_max', 0.85)
        self.min_ball_area = detection_cfg.get('min_ball_area', 35)
        self.max_ball_area = detection_cfg.get('max_ball_area', 3200)
        self.ball_aspect_ratio_min = detection_cfg.get('ball_aspect_ratio_min', 0.55)
        self.ball_aspect_ratio_max = detection_cfg.get('ball_aspect_ratio_max', 1.7)
        self.min_visible_keypoints = detection_cfg.get('min_visible_keypoints', 5)
        self.player_nms_iou = detection_cfg.get('player_nms_iou', 0.5)
        self.pose_player_iou = detection_cfg.get('pose_player_iou', 0.55)
        
        jersey_cfg = detection_cfg.get('jersey', {})
        self.jersey_saturation_min = jersey_cfg.get('saturation_min', 45)
        self.jersey_value_min = jersey_cfg.get('value_min', 35)
        self.jersey_value_max = jersey_cfg.get('value_max', 245)
        self.grass_h_min = jersey_cfg.get('exclude_grass_h_min', 30)
        self.grass_h_max = jersey_cfg.get('exclude_grass_h_max', 95)
        
        self.max_referee_ratio = detection_cfg.get('max_referee_ratio', 0.25)
        self.team_prototype_alpha = detection_cfg.get('team_prototype_alpha', 0.2)
        
        # Historique de couleurs pour stabiliser l'assignation des équipes
        self.team_color_prototypes = {'team_1': None, 'team_2': None}
        
        # Indices LAB optionnels depuis la config couleurs
        self.team_color_hints = {}
        for team_name in ('team_1', 'team_2'):
            bgr = self.config.get('colors', {}).get(team_name)
            if isinstance(bgr, list) and len(bgr) == 3:
                lab = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
                self.team_color_hints[team_name] = lab
        
        print(f"✅ Détecteur professionnel initialisé")
    
    def _setup_gpu(self) -> str:
        """Configure le GPU"""
        if self.config['gpu']['enabled'] and torch.cuda.is_available():
            device = self.config['gpu']['device']
            print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
            return device
        return "cpu"
    
    def detect_with_pose(self, frame: np.ndarray) -> Tuple[List, List, List]:
        """
        Détection avancée avec pose estimation
        
        Returns:
            players: Détections de joueurs avec poses
            balls: Détections de ballon
            keypoints: Points clés des poses
        """
        h = frame.shape[0]
        
        # 1. Détection des objets (joueurs + ballon)
        base_conf = min(self.player_conf_threshold, self.ball_conf_threshold)
        det_results = self.detector(
            frame,
            conf=base_conf,
            iou=self.iou_threshold,
            imgsz=self.img_size,
            device=self.device,
            verbose=False,
            classes=[0, 32]  # Personnes et ballons uniquement
        )
        
        # 2. Détection des poses
        pose_results = self.pose_model(
            frame,
            conf=self.pose_conf_threshold,
            device=self.device,
            verbose=False
        )
        
        players = []
        balls = []
        keypoints_all = []
        pose_player_candidates = []
        
        # Parser les détections
        for result in det_results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                
                if cls == 0:  # Personne
                    if conf >= self.player_conf_threshold and self._is_valid_player_box(
                        x1, y1, x2, y2, h
                    ):
                        players.append({
                            'bbox': [x1, y1, x2, y2],
                            'conf': conf,
                            'class': cls,
                            'area': (x2 - x1) * (y2 - y1),
                            'center': ((x1 + x2) / 2, (y1 + y2) / 2)
                        })
                
                elif cls == 32:  # Ballon
                    if conf >= self.ball_conf_threshold and self._is_valid_ball_box(
                        x1, y1, x2, y2, h
                    ):
                        balls.append({
                            'bbox': [x1, y1, x2, y2],
                            'conf': conf,
                            'class': cls
                        })
        
        # Parser les poses
        for pose_result in pose_results:
            if hasattr(pose_result, 'keypoints') and pose_result.keypoints is not None:
                kpts = pose_result.keypoints.data.cpu().numpy()
                pose_boxes = pose_result.boxes
                
                for idx, person_kpts in enumerate(kpts):
                    # person_kpts shape: (17, 3) - 17 keypoints avec (x, y, conf)
                    keypoints_all.append(person_kpts)
                    
                    if pose_boxes is None or idx >= len(pose_boxes):
                        continue
                    
                    box = pose_boxes[idx]
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    visible_count = int(np.sum(person_kpts[:, 2] > 0.3))
                    
                    if conf < self.pose_conf_threshold or visible_count < self.min_visible_keypoints:
                        continue
                    
                    if not self._is_valid_player_box(x1, y1, x2, y2, h):
                        continue
                    
                    pose_player_candidates.append({
                        'bbox': [x1, y1, x2, y2],
                        'conf': conf,
                        'class': 0,
                        'area': (x2 - x1) * (y2 - y1),
                        'center': ((x1 + x2) / 2, (y1 + y2) / 2),
                        'keypoints': person_kpts
                    })
        
        # Fusionner les boîtes de pose non couvertes par le détecteur
        for pose_player in pose_player_candidates:
            has_overlap = any(
                self._calculate_iou(pose_player['bbox'], detected['bbox']) >= self.pose_player_iou
                for detected in players
            )
            if not has_overlap:
                players.append(pose_player)
        
        # Associer poses aux joueurs détectés
        players = self._match_poses_to_players(players, keypoints_all)
        
        # Éliminer les doublons (même joueur détecté plusieurs fois)
        players = self._remove_duplicate_detections(players)
        
        return players, balls, keypoints_all

    def _is_valid_player_box(self, x1: float, y1: float, x2: float, y2: float, frame_h: int) -> bool:
        """Valide une bbox joueur avec contraintes géométriques."""
        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0:
            return False
        
        area = width * height
        aspect_ratio = height / width
        center_y = (y1 + y2) / 2
        
        return (
            self.min_player_area <= area <= self.max_player_area and
            self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio and
            frame_h * self.field_y_min <= center_y <= frame_h * self.field_y_max
        )

    def _is_valid_ball_box(self, x1: float, y1: float, x2: float, y2: float, frame_h: int) -> bool:
        """Valide une bbox ballon sans dégrader la précision."""
        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0:
            return False
        
        area = width * height
        aspect_ratio = height / width
        center_y = (y1 + y2) / 2
        
        return (
            self.min_ball_area <= area <= self.max_ball_area and
            self.ball_aspect_ratio_min <= aspect_ratio <= self.ball_aspect_ratio_max and
            frame_h * self.ball_field_y_min <= center_y <= frame_h * self.ball_field_y_max
        )
    
    def _match_poses_to_players(self, players: List[Dict], keypoints: List) -> List[Dict]:
        """Associe les poses détectées aux bounding boxes des joueurs"""
        for player in players:
            bbox = player['bbox']
            x1, y1, x2, y2 = bbox
            
            # Trouver la pose la plus proche
            best_match = None
            best_score = -1.0
            
            for kpts in keypoints:
                visible_kpts = kpts[kpts[:, 2] > 0.3]  # Confiance > 0.3
                if len(visible_kpts) > 0:
                    pose_center_x = np.mean(visible_kpts[:, 0])
                    pose_center_y = np.mean(visible_kpts[:, 1])
                    
                    if x1 <= pose_center_x <= x2 and y1 <= pose_center_y <= y2:
                        # Favorise les poses riches en points visibles et proches du centre
                        box_center_x = (x1 + x2) / 2.0
                        box_center_y = (y1 + y2) / 2.0
                        distance = np.hypot(pose_center_x - box_center_x, pose_center_y - box_center_y)
                        score = len(visible_kpts) - 0.02 * distance
                        if score > best_score:
                            best_score = score
                            best_match = kpts
            
            player['keypoints'] = best_match
        
        return players
    
    def _remove_duplicate_detections(self, players: List[Dict]) -> List[Dict]:
        """Élimine les détections en double (NMS personnalisé)"""
        if len(players) <= 1:
            return players
        
        # Trier par confiance décroissante
        players_sorted = sorted(players, key=lambda x: x['conf'], reverse=True)
        
        keep = []
        while players_sorted:
            # Prendre le meilleur
            best = players_sorted.pop(0)
            keep.append(best)
            
            # Éliminer ceux qui se chevauchent trop
            players_sorted = [
                p for p in players_sorted
                if self._calculate_iou(best['bbox'], p['bbox']) < self.player_nms_iou
            ]
        
        return keep
    
    def _calculate_iou(self, box1: List, box2: List) -> float:
        """Calcule l'IoU entre deux boxes"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Intersection
        xi_min = max(x1_min, x2_min)
        yi_min = max(y1_min, y2_min)
        xi_max = min(x1_max, x2_max)
        yi_max = min(y1_max, y2_max)
        
        if xi_max <= xi_min or yi_max <= yi_min:
            return 0.0
        
        intersection = (xi_max - xi_min) * (yi_max - yi_min)
        
        # Union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _extract_jersey_feature_lab(self, frame: np.ndarray, player: Dict) -> Optional[np.ndarray]:
        """Extrait une signature couleur robuste du maillot (espace LAB)."""
        frame_h, frame_w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, player['bbox'])
        
        x1 = max(0, min(frame_w - 1, x1))
        y1 = max(0, min(frame_h - 1, y1))
        x2 = max(0, min(frame_w, x2))
        y2 = max(0, min(frame_h, y2))
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        box_h = y2 - y1
        box_w = x2 - x1
        torso_y1 = y1 + int(box_h * 0.16)
        torso_y2 = y1 + int(box_h * 0.55)
        torso_x1 = x1 + int(box_w * 0.20)
        torso_x2 = x2 - int(box_w * 0.20)
        
        torso = frame[torso_y1:torso_y2, torso_x1:torso_x2]
        if torso.size == 0:
            return None
        
        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(torso, cv2.COLOR_BGR2LAB)
        
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        hue = hsv[:, :, 0]
        
        valid_light = (value >= self.jersey_value_min) & (value <= self.jersey_value_max)
        valid_saturation = saturation >= self.jersey_saturation_min
        non_grass = (hue < self.grass_h_min) | (hue > self.grass_h_max)
        jersey_mask = valid_light & valid_saturation & non_grass
        
        if np.sum(jersey_mask) < 40:
            # Fallback doux si le masque est trop agressif
            lab_pixels = lab.reshape(-1, 3)
        else:
            lab_pixels = lab[jersey_mask]
        
        if len(lab_pixels) == 0:
            return None
        
        return np.median(lab_pixels, axis=0).astype(np.float32)

    def _map_team_clusters(self, centers: Dict[int, np.ndarray]) -> Dict[int, str]:
        """Assigne les clusters aux labels team_1/team_2 en conservant la stabilité temporelle."""
        cluster_ids = list(centers.keys())
        if len(cluster_ids) < 2:
            return {}
        
        c0, c1 = cluster_ids[0], cluster_ids[1]
        center0 = centers[c0]
        center1 = centers[c1]
        
        proto1 = self.team_color_prototypes['team_1']
        proto2 = self.team_color_prototypes['team_2']
        
        if proto1 is not None and proto2 is not None:
            same_cost = np.linalg.norm(center0 - proto1) + np.linalg.norm(center1 - proto2)
            swap_cost = np.linalg.norm(center0 - proto2) + np.linalg.norm(center1 - proto1)
            if same_cost <= swap_cost:
                return {c0: 'team_1', c1: 'team_2'}
            return {c0: 'team_2', c1: 'team_1'}
        
        # Initialisation guidée par des indices couleur configurés, si disponibles.
        hint1 = self.team_color_hints.get('team_1')
        hint2 = self.team_color_hints.get('team_2')
        if hint1 is not None and hint2 is not None:
            same_cost = np.linalg.norm(center0 - hint1) + np.linalg.norm(center1 - hint2)
            swap_cost = np.linalg.norm(center0 - hint2) + np.linalg.norm(center1 - hint1)
            if same_cost <= swap_cost:
                return {c0: 'team_1', c1: 'team_2'}
            return {c0: 'team_2', c1: 'team_1'}
        
        # Dernier fallback déterministe.
        ordered = sorted(cluster_ids)
        return {ordered[0]: 'team_1', ordered[1]: 'team_2'}

    def _update_team_prototypes(self, team_centers: Dict[str, np.ndarray]) -> None:
        """Met à jour les prototypes couleur (EMA) pour stabiliser les frames suivantes."""
        for team_name, center in team_centers.items():
            if center is None:
                continue
            previous = self.team_color_prototypes[team_name]
            if previous is None:
                self.team_color_prototypes[team_name] = center.astype(np.float32)
            else:
                alpha = self.team_prototype_alpha
                self.team_color_prototypes[team_name] = (1 - alpha) * previous + alpha * center

    def classify_teams_advanced(self, frame: np.ndarray, players: List[Dict]) -> Dict:
        """
        Classification avancée avec cohérence temporelle inter-frames.
        """
        if not players:
            return {'team_1': [], 'team_2': [], 'referee': []}
        
        features = []
        feature_players = []
        missing_feature_players = []
        
        for player in players:
            feature = self._extract_jersey_feature_lab(frame, player)
            if feature is None:
                missing_feature_players.append(player)
                continue
            features.append(feature)
            feature_players.append(player)
        
        result = {'team_1': [], 'team_2': [], 'referee': []}
        if len(features) < 2:
            # Fallback: conserver tous les joueurs en jeu, même sans signature couleur fiable.
            for idx, player in enumerate(players):
                target = 'team_1' if idx % 2 == 0 else 'team_2'
                result[target].append(player)
            return result
        
        from sklearn.cluster import KMeans
        
        colors_array = np.array(features, dtype=np.float32)
        n_clusters = 3 if len(colors_array) >= 6 else 2
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(colors_array)
        
        cluster_counts = {i: int(np.sum(labels == i)) for i in range(n_clusters)}
        sorted_clusters = sorted(cluster_counts.keys(), key=lambda cid: cluster_counts[cid], reverse=True)
        
        referee_cluster = -1
        if n_clusters == 3:
            candidate = min(cluster_counts, key=cluster_counts.get)
            max_referee_players = max(
                1, min(2, int(len(feature_players) * self.max_referee_ratio))
            )
            if cluster_counts[candidate] <= max_referee_players:
                referee_cluster = candidate
        
        team_clusters = [cid for cid in sorted_clusters if cid != referee_cluster][:2]
        if len(team_clusters) < 2:
            # Séparation minimale quand KMeans ne permet pas de distinguer deux groupes stables.
            for idx, player in enumerate(players):
                target = 'team_1' if idx % 2 == 0 else 'team_2'
                result[target].append(player)
            return result
        
        centers = {cid: kmeans.cluster_centers_[cid].astype(np.float32) for cid in team_clusters}
        cluster_to_team = self._map_team_clusters(centers)
        
        for player, label in zip(feature_players, labels):
            if label == referee_cluster:
                result['referee'].append(player)
                continue
            team_name = cluster_to_team.get(label)
            if team_name is None:
                # Cluster résiduel: rattacher à l'équipe la plus proche des prototypes.
                center = kmeans.cluster_centers_[label].astype(np.float32)
                proto1 = self.team_color_prototypes['team_1']
                proto2 = self.team_color_prototypes['team_2']
                if proto1 is None or proto2 is None:
                    team_name = 'team_1' if len(result['team_1']) <= len(result['team_2']) else 'team_2'
                else:
                    d1 = np.linalg.norm(center - proto1)
                    d2 = np.linalg.norm(center - proto2)
                    team_name = 'team_1' if d1 <= d2 else 'team_2'
            result[team_name].append(player)
        
        # Ne pas perdre les joueurs sans feature: assignation spatiale à l'équipe la plus proche.
        for player in missing_feature_players:
            p_center = np.array(player['center'], dtype=np.float32)
            team1_centers = [np.array(p['center'], dtype=np.float32) for p in result['team_1']]
            team2_centers = [np.array(p['center'], dtype=np.float32) for p in result['team_2']]
            
            if team1_centers and team2_centers:
                d1 = min(np.linalg.norm(p_center - c) for c in team1_centers)
                d2 = min(np.linalg.norm(p_center - c) for c in team2_centers)
                team_name = 'team_1' if d1 <= d2 else 'team_2'
            elif team1_centers:
                team_name = 'team_1'
            elif team2_centers:
                team_name = 'team_2'
            else:
                team_name = 'team_1' if len(result['team_1']) <= len(result['team_2']) else 'team_2'
            result[team_name].append(player)
        
        self._update_team_prototypes({
            'team_1': centers.get(next((cid for cid, team in cluster_to_team.items() if team == 'team_1'), -1)),
            'team_2': centers.get(next((cid for cid, team in cluster_to_team.items() if team == 'team_2'), -1))
        })
        
        return result


if __name__ == "__main__":
    # Test
    detector = ProfessionalFootballDetector()
    print("✅ Détecteur professionnel prêt")
