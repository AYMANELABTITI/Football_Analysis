"""
Visualiseur professionnel FIFA-style
Interface propre avec skeleton tracking et heatmaps
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from collections import defaultdict


class ProfessionalVisualizer:
    """Visualiseur de qualité FIFA avec skeleton tracking"""
    
    def __init__(self):
        """Initialise le visualiseur professionnel"""
        
        # Couleurs professionnelles (BGR)
        self.colors = {
            'team_1': (50, 180, 255),    # Orange vif
            'team_2': (255, 180, 50),    # Bleu cyan
            'referee': (200, 200, 200),  # Gris clair
            'ball': (0, 255, 255),       # Jaune vif
            'skeleton': (255, 255, 255), # Blanc pour skeleton
            'text_bg': (40, 40, 40),     # Fond texte sombre
            'text': (255, 255, 255)      # Texte blanc
        }
        
        # Connexions du skeleton COCO (17 keypoints)
        self.skeleton_connections = [
            (0, 1), (0, 2),  # Tête
            (1, 3), (2, 4),  # Bras supérieurs
            (5, 6),  # Épaules
            (5, 7), (7, 9),  # Bras gauche
            (6, 8), (8, 10),  # Bras droit
            (5, 11), (6, 12),  # Torse
            (11, 12),  # Hanches
            (11, 13), (13, 15),  # Jambe gauche
            (12, 14), (14, 16)   # Jambe droite
        ]
        
        # Historiques
        self.player_history = defaultdict(list)
        self.ball_history = []
        
        print("✅ Visualiseur professionnel initialisé")
    
    def draw_clean_player(self, frame: np.ndarray, player: Dict, 
                         color: Tuple[int, int, int], player_id: int = None,
                         show_skeleton: bool = True) -> np.ndarray:
        """
        Dessine un joueur de manière professionnelle (FIFA-style)
        
        Args:
            frame: Image
            player: Données du joueur
            color: Couleur de l'équipe
            player_id: ID du joueur
            show_skeleton: Afficher le skeleton ou juste la box
        """
        x1, y1, x2, y2 = map(int, player['bbox'])
        conf = player['conf']
        
        # 1. Dessiner le skeleton si disponible
        if show_skeleton and 'keypoints' in player and player['keypoints'] is not None:
            keypoints = player['keypoints']
            self._draw_skeleton(frame, keypoints, color)
        
        # 2. Bounding box fine et propre
        # Box externe (couleur d'équipe)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # 3. ID et confiance - STYLE FIFA
        if player_id is not None:
            # Position du texte (au-dessus de la tête)
            text = f"ID: {player_id}"
            conf_text = f"{conf:.2f}"
            
            # Taille du texte
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            
            # Calculer la taille du fond
            (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
            (conf_w, conf_h), _ = cv2.getTextSize(conf_text, font, font_scale, thickness)
            
            # Position (centré au-dessus de la bbox)
            center_x = int((x1 + x2) / 2)
            text_x = center_x - text_w // 2
            text_y = y1 - 15
            
            # Fond semi-transparent pour le texte
            overlay = frame.copy()
            padding = 4
            cv2.rectangle(overlay,
                         (text_x - padding, text_y - text_h - padding),
                         (text_x + text_w + padding, text_y + padding),
                         self.colors['text_bg'], -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Texte ID (couleur d'équipe)
            cv2.putText(frame, text, (text_x, text_y),
                       font, font_scale, color, thickness + 1)
            
            # Confiance en petit (blanc)
            conf_x = center_x - conf_w // 2
            cv2.putText(frame, conf_text, (conf_x, text_y - text_h - 3),
                       font, 0.35, (180, 180, 180), 1)
        
        # 4. Point central (petit cercle)
        center = player['center']
        cv2.circle(frame, (int(center[0]), int(center[1])), 4, color, -1)
        cv2.circle(frame, (int(center[0]), int(center[1])), 5, (255, 255, 255), 1)
        
        return frame
    
    def _draw_skeleton(self, frame: np.ndarray, keypoints: np.ndarray,
                      color: Tuple[int, int, int]):
        """Dessine le skeleton d'un joueur (style FIFA)"""
        if keypoints is None:
            return
        
        # Dessiner les connexions
        for connection in self.skeleton_connections:
            pt1_idx, pt2_idx = connection
            
            if pt1_idx >= len(keypoints) or pt2_idx >= len(keypoints):
                continue
            
            pt1 = keypoints[pt1_idx]
            pt2 = keypoints[pt2_idx]
            
            # Vérifier que les deux points sont visibles (conf > 0.3)
            if pt1[2] > 0.3 and pt2[2] > 0.3:
                x1, y1 = int(pt1[0]), int(pt1[1])
                x2, y2 = int(pt2[0]), int(pt2[1])
                
                # Ligne du skeleton (couleur d'équipe)
                cv2.line(frame, (x1, y1), (x2, y2), color, 2)
        
        # Dessiner les keypoints
        for i, kpt in enumerate(keypoints):
            if kpt[2] > 0.3:  # Confiance > 0.3
                x, y = int(kpt[0]), int(kpt[1])
                
                # Cercle pour chaque keypoint
                cv2.circle(frame, (x, y), 3, color, -1)
                cv2.circle(frame, (x, y), 4, (255, 255, 255), 1)
    
    def draw_clean_ball(self, frame: np.ndarray, ball: Dict) -> np.ndarray:
        """Dessine le ballon de manière professionnelle"""
        x1, y1, x2, y2 = map(int, ball['bbox'])
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        
        # Cercle extérieur blanc
        cv2.circle(frame, (center_x, center_y), 12, (255, 255, 255), 2)
        
        # Cercle intérieur jaune
        cv2.circle(frame, (center_x, center_y), 8, self.colors['ball'], -1)
        
        # Icône ballon (petit)
        cv2.circle(frame, (center_x, center_y), 3, (0, 0, 0), -1)
        
        return frame
    
    def draw_professional_hud(self, frame: np.ndarray, stats: Dict) -> np.ndarray:
        """
        Dessine un HUD professionnel en haut de l'écran
        
        Args:
            frame: Image
            stats: Statistiques à afficher
        """
        h, w = frame.shape[:2]
        
        # Barre supérieure semi-transparente
        overlay = frame.copy()
        hud_height = 80
        cv2.rectangle(overlay, (0, 0), (w, hud_height), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Ligne de séparation
        cv2.line(frame, (0, hud_height), (w, hud_height), (100, 100, 100), 2)
        
        # Titre
        cv2.putText(frame, "FOOTBALL ANALYSIS - PROFESSIONAL MODE",
                   (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Stats (3 colonnes)
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_pos = 60
        
        # Colonne 1: Joueurs
        team1_count = stats.get('team_1_count', 0)
        team2_count = stats.get('team_2_count', 0)
        cv2.putText(frame, f"Team 1: {team1_count}", (20, y_pos),
                   font, 0.5, self.colors['team_1'], 1)
        cv2.putText(frame, f"Team 2: {team2_count}", (150, y_pos),
                   font, 0.5, self.colors['team_2'], 1)
        
        # Colonne 2: Frame info
        frame_num = stats.get('frame', 0)
        total_frames = stats.get('total_frames', 0)
        cv2.putText(frame, f"Frame: {frame_num}/{total_frames}",
                   (w // 2 - 80, y_pos), font, 0.5, (200, 200, 200), 1)
        
        # Colonne 3: FPS / Events
        fps = stats.get('fps', 0)
        events = stats.get('events', 0)
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 200, y_pos),
                   font, 0.5, (100, 255, 100), 1)
        cv2.putText(frame, f"Events: {events}", (w - 100, y_pos),
                   font, 0.5, (255, 255, 100), 1)
        
        return frame
    
    def create_clean_heatmap(self, positions_dict: Dict[str, List[Tuple]],
                            field_size: Tuple[int, int] = (1920, 1080)) -> np.ndarray:
        """
        Crée une heatmap propre style FIFA
        
        Args:
            positions_dict: {'team_1': [...], 'team_2': [...]}
            field_size: Dimensions du terrain
        """
        w, h = field_size
        
        # Créer deux heatmaps séparées
        heatmap1 = np.zeros((h, w), dtype=np.float32)
        heatmap2 = np.zeros((h, w), dtype=np.float32)
        
        # Team 1 (Rouge)
        for pos in positions_dict.get('team_1', []):
            x, y = int(pos[0]), int(pos[1])
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(heatmap1, (x, y), 40, 1.0, -1)
        
        # Team 2 (Bleu)
        for pos in positions_dict.get('team_2', []):
            x, y = int(pos[0]), int(pos[1])
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(heatmap2, (x, y), 40, 1.0, -1)
        
        # Appliquer flou gaussien
        heatmap1 = cv2.GaussianBlur(heatmap1, (0, 0), 25)
        heatmap2 = cv2.GaussianBlur(heatmap2, (0, 0), 25)
        
        # Normaliser
        if heatmap1.max() > 0:
            heatmap1 = heatmap1 / heatmap1.max()
        if heatmap2.max() > 0:
            heatmap2 = heatmap2 / heatmap2.max()
        
        # Créer l'image RGB
        heatmap_img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Team 1 en rouge
        heatmap_img[:, :, 2] = (heatmap1 * 255).astype(np.uint8)
        
        # Team 2 en bleu
        heatmap_img[:, :, 0] = (heatmap2 * 255).astype(np.uint8)
        
        # Dessiner les lignes du terrain
        heatmap_img = self._draw_field_lines(heatmap_img)
        
        return heatmap_img
    
    def _draw_field_lines(self, img: np.ndarray) -> np.ndarray:
        """Dessine les lignes du terrain sur une image"""
        # OpenCV exige un tableau C-contigu
        img = np.ascontiguousarray(img)
        h, w = img.shape[:2]
        
        white = (255, 255, 255)
        thickness = 2
        
        # Contour
        cv2.rectangle(img, (50, 50), (w - 50, h - 50), white, thickness)
        
        # Ligne médiane
        cv2.line(img, (w // 2, 50), (w // 2, h - 50), white, thickness)
        
        # Cercle central
        cv2.circle(img, (w // 2, h // 2), 80, white, thickness)
        
        # Surfaces de réparation
        penalty_w = 180
        penalty_h = 300
        
        cv2.rectangle(img, (50, h // 2 - penalty_h // 2),
                     (50 + penalty_w, h // 2 + penalty_h // 2), white, thickness)
        cv2.rectangle(img, (w - 50 - penalty_w, h // 2 - penalty_h // 2),
                     (w - 50, h // 2 + penalty_h // 2), white, thickness)
        
        return img
    
    def create_pass_map(self, passes: List[Dict],
                       size: Tuple[int, int] = (1920, 1080)) -> np.ndarray:
        """
        Crée une visualisation graphique des passes sur un terrain.
        
        Args:
            passes: Liste des passes détectées
            size: Taille de l'image (width, height)
        """
        w, h = size
        pass_map = np.full((h, w, 3), (34, 139, 34), dtype=np.uint8)
        pass_map = self._draw_field_lines(pass_map)
        
        if not passes:
            cv2.putText(pass_map, "No pass detected", (60, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            return pass_map
        
        overlay = pass_map.copy()
        team_1_count = 0
        team_2_count = 0
        
        for pass_event in passes:
            if 'start_x' not in pass_event or 'end_x' not in pass_event:
                continue
            
            start = (int(pass_event['start_x']), int(pass_event['start_y']))
            end = (int(pass_event['end_x']), int(pass_event['end_y']))
            team = pass_event.get('team', 'team_1')
            color = self.colors.get(team, (255, 255, 255))
            
            if team == 'team_1':
                team_1_count += 1
            elif team == 'team_2':
                team_2_count += 1
            
            cv2.arrowedLine(overlay, start, end, color, 2, tipLength=0.22)
            cv2.circle(overlay, start, 4, color, -1)
            cv2.circle(overlay, end, 5, (255, 255, 255), -1)
        
        cv2.addWeighted(overlay, 0.70, pass_map, 0.30, 0, pass_map)
        
        info_overlay = pass_map.copy()
        cv2.rectangle(info_overlay, (20, 20), (560, 110), (0, 0, 0), -1)
        cv2.addWeighted(info_overlay, 0.65, pass_map, 0.35, 0, pass_map)
        
        total = team_1_count + team_2_count
        cv2.putText(pass_map, f"Pass map - Total: {total}", (35, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(pass_map, f"Team 1: {team_1_count}", (35, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['team_1'], 2)
        cv2.putText(pass_map, f"Team 2: {team_2_count}", (260, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['team_2'], 2)
        
        return pass_map
    
    def create_mini_map(self, players_team1: List, players_team2: List,
                       ball_pos: Tuple = None, size: Tuple[int, int] = (400, 300)) -> np.ndarray:
        """Crée une mini-map tactique (vue de dessus)"""
        w, h = size
        
        # Fond vert terrain
        minimap = np.full((h, w, 3), (34, 139, 34), dtype=np.uint8)
        
        # Lignes du terrain
        minimap = self._draw_field_lines(minimap)
        
        # Dessiner les joueurs (points)
        for player in players_team1:
            cx, cy = player['center']
            # Normaliser position
            x = int((cx / 1920) * w)
            y = int((cy / 1080) * h)
            cv2.circle(minimap, (x, y), 6, self.colors['team_1'], -1)
            cv2.circle(minimap, (x, y), 7, (255, 255, 255), 1)
        
        for player in players_team2:
            cx, cy = player['center']
            x = int((cx / 1920) * w)
            y = int((cy / 1080) * h)
            cv2.circle(minimap, (x, y), 6, self.colors['team_2'], -1)
            cv2.circle(minimap, (x, y), 7, (255, 255, 255), 1)
        
        # Ballon
        if ball_pos:
            bx, by = ball_pos
            x = int((bx / 1920) * w)
            y = int((by / 1080) * h)
            cv2.circle(minimap, (x, y), 8, self.colors['ball'], -1)
            cv2.circle(minimap, (x, y), 9, (0, 0, 0), 2)
        
        return minimap


if __name__ == "__main__":
    viz = ProfessionalVisualizer()
    print("✅ Visualiseur professionnel prêt")
