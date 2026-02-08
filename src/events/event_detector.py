"""
Module de détection d'événements clés du match
Détecte: hors-jeu, corner, penalty, fautes, passes, tirs
"""

import numpy as np
from typing import List, Dict, Tuple
import yaml
from collections import defaultdict


class EventDetector:
    """
    Détecteur d'événements dans un match de football
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialise le détecteur d'événements
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Dimensions du terrain
        self.field_length = self.config['field']['length']
        self.field_width = self.config['field']['width']
        self.penalty_area_length = self.config['field']['penalty_area_length']
        
        # Seuils
        self.offside_threshold = self.config['events']['offside_threshold']
        self.possession_distance = self.config['events']['ball_possession_distance']
        self.high_speed = self.config['events']['high_speed_threshold']
        self.sprint_speed = self.config['events']['sprint_threshold']
        
        # Historique des événements
        self.events = []
        self.ball_history = []
        self.player_speeds = defaultdict(list)
        
        print(f"✅ Détecteur d'événements initialisé")
    
    def set_field_homography(self, homography_matrix: np.ndarray):
        """
        Définit la matrice d'homographie pour transformer pixels -> coordonnées terrain
        """
        self.homography = homography_matrix
    
    def pixel_to_field(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """
        Convertit des coordonnées pixel en coordonnées terrain (mètres)
        """
        if not hasattr(self, 'homography'):
            # Si pas d'homographie, retourner des coordonnées normalisées
            return point[0] / 10, point[1] / 10
        
        # Appliquer la transformation d'homographie
        pt = np.array([point[0], point[1], 1.0])
        transformed = self.homography @ pt
        if transformed[2] != 0:
            return transformed[0] / transformed[2], transformed[1] / transformed[2]
        return point[0], point[1]
    
    def calculate_speed(self, positions: List[Tuple], fps: int = 30) -> float:
        """
        Calcule la vitesse en km/h à partir des positions
        
        Args:
            positions: Liste de positions [(x, y), ...]
            fps: Frames par seconde
            
        Returns:
            Vitesse en km/h
        """
        if len(positions) < 2:
            return 0.0
        
        # Calculer la distance entre dernières positions
        p1 = self.pixel_to_field(positions[-2])
        p2 = self.pixel_to_field(positions[-1])
        
        distance_m = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        
        # Convertir en km/h
        time_s = 1.0 / fps
        speed_kmh = (distance_m / time_s) * 3.6
        
        return speed_kmh
    
    def detect_ball_possession(self, ball_pos: Tuple, players: List[Dict]) -> Dict:
        """
        Détecte quel joueur a la possession du ballon
        
        Args:
            ball_pos: Position du ballon (x, y)
            players: Liste des joueurs trackés
            
        Returns:
            Joueur en possession ou None
        """
        min_distance = float('inf')
        possessor = None
        
        for player in players:
            # Calculer la distance entre joueur et ballon
            player_center = (
                (player['bbox'][0] + player['bbox'][2]) / 2,
                (player['bbox'][1] + player['bbox'][3]) / 2
            )
            
            distance = np.sqrt(
                (ball_pos[0] - player_center[0])**2 +
                (ball_pos[1] - player_center[1])**2
            )
            
            if distance < min_distance:
                min_distance = distance
                possessor = player
        
        # Vérifier si assez proche pour avoir possession
        if min_distance < self.possession_distance * 100:  # Conversion approximative
            return possessor
        
        return None
    
    def detect_offside(self, team_1_players: List[Dict], team_2_players: List[Dict],
                      ball_pos: Tuple, attacking_team: int) -> List[Dict]:
        """
        Détecte les situations de hors-jeu
        
        Args:
            team_1_players: Joueurs équipe 1
            team_2_players: Joueurs équipe 2
            ball_pos: Position du ballon
            attacking_team: 1 ou 2
            
        Returns:
            Liste des joueurs en position de hors-jeu
        """
        if attacking_team == 1:
            attacking = team_1_players
            defending = team_2_players
        else:
            attacking = team_2_players
            defending = team_1_players
        
        offside_players = []
        
        if len(defending) < 2:
            return offside_players
        
        # Trouver les deux défenseurs les plus avancés
        defender_positions = []
        for defender in defending:
            center_x = (defender['bbox'][0] + defender['bbox'][2]) / 2
            defender_positions.append(center_x)
        
        defender_positions.sort()
        second_last_defender = defender_positions[-2] if len(defender_positions) >= 2 else defender_positions[-1]
        
        # Vérifier chaque attaquant
        for attacker in attacking:
            attacker_x = (attacker['bbox'][0] + attacker['bbox'][2]) / 2
            
            # Si attaquant est devant le second dernier défenseur et devant le ballon
            if attacker_x > second_last_defender and attacker_x > ball_pos[0]:
                offside_players.append({
                    'player_id': attacker['id'],
                    'position': attacker_x,
                    'offside_line': second_last_defender
                })
        
        return offside_players
    
    def detect_in_penalty_area(self, position: Tuple, side: str = 'left') -> bool:
        """
        Détecte si une position est dans la surface de réparation
        
        Args:
            position: (x, y) en coordonnées terrain
            side: 'left' ou 'right'
            
        Returns:
            True si dans la surface
        """
        x, y = self.pixel_to_field(position)
        
        # Surface gauche
        if side == 'left':
            if x <= self.penalty_area_length and abs(y - self.field_width/2) <= self.penalty_area_length:
                return True
        # Surface droite
        else:
            if x >= (self.field_length - self.penalty_area_length) and \
               abs(y - self.field_width/2) <= self.penalty_area_length:
                return True
        
        return False
    
    def detect_corner(self, ball_pos: Tuple) -> Dict:
        """
        Détecte si le ballon est dans une zone de corner
        """
        x, y = self.pixel_to_field(ball_pos)
        
        corner_threshold = 5  # mètres du coin
        
        # Coins
        corners = [
            ('bottom_left', 0, 0),
            ('bottom_right', 0, self.field_width),
            ('top_left', self.field_length, 0),
            ('top_right', self.field_length, self.field_width)
        ]
        
        for corner_name, cx, cy in corners:
            distance = np.sqrt((x - cx)**2 + (y - cy)**2)
            if distance < corner_threshold:
                return {
                    'type': 'corner',
                    'corner': corner_name,
                    'position': ball_pos
                }
        
        return None
    
    def detect_shot(self, ball_trajectory: List[Tuple], goal_area: Tuple) -> bool:
        """
        Détecte un tir au but basé sur la trajectoire du ballon
        """
        if len(ball_trajectory) < 5:
            return False
        
        # Calculer la vitesse du ballon
        recent_positions = ball_trajectory[-5:]
        speed = 0
        
        for i in range(1, len(recent_positions)):
            p1 = self.pixel_to_field(recent_positions[i-1])
            p2 = self.pixel_to_field(recent_positions[i])
            dist = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            speed += dist
        
        # Si vitesse élevée et direction vers le but
        if speed > 10:  # Seuil de vitesse
            last_pos = self.pixel_to_field(ball_trajectory[-1])
            # Vérifier si se dirige vers une zone de but
            if self.detect_in_penalty_area(ball_trajectory[-1], 'left') or \
               self.detect_in_penalty_area(ball_trajectory[-1], 'right'):
                return True
        
        return False
    
    def detect_pass(self, ball_history: List[Dict], possession_history: List[Dict]) -> Dict:
        """
        Détecte une passe entre joueurs
        """
        if len(possession_history) < 2:
            return None
        
        # Vérifier changement de possession
        if possession_history[-1] != possession_history[-2]:
            return {
                'type': 'pass',
                'from_player': possession_history[-2]['id'] if possession_history[-2] else None,
                'to_player': possession_history[-1]['id'] if possession_history[-1] else None,
                'success': True
            }
        
        return None
    
    def analyze_player_movement(self, player_tracks: Dict) -> Dict:
        """
        Analyse les mouvements des joueurs (sprints, zones d'activité)
        """
        analysis = {}
        
        for player_id, track in player_tracks.items():
            if len(track['positions']) < 10:
                continue
            
            # Calculer vitesse
            speed = self.calculate_speed(track['positions'][-10:])
            
            # Détecter sprint
            is_sprinting = speed > self.sprint_speed
            
            # Calculer distance totale
            total_distance = track['distance']
            
            # Zone d'activité (heatmap)
            positions_field = [self.pixel_to_field(p) for p in track['positions']]
            
            analysis[player_id] = {
                'current_speed': speed,
                'is_sprinting': is_sprinting,
                'total_distance': total_distance,
                'positions': positions_field,
                'avg_speed': np.mean(self.player_speeds[player_id]) if self.player_speeds[player_id] else 0
            }
            
            self.player_speeds[player_id].append(speed)
        
        return analysis
    
    def process_frame(self, frame_data: Dict) -> List[Dict]:
        """
        Traite une frame et détecte tous les événements
        
        Args:
            frame_data: {
                'ball': position ou None,
                'team_1': liste de joueurs,
                'team_2': liste de joueurs,
                'frame_id': numéro de frame
            }
            
        Returns:
            Liste d'événements détectés
        """
        events = []
        
        # Détecter possession
        if frame_data['ball'] is not None:
            all_players = frame_data['team_1'] + frame_data['team_2']
            possessor = self.detect_ball_possession(frame_data['ball'], all_players)
            
            if possessor:
                events.append({
                    'type': 'possession',
                    'player_id': possessor['id'],
                    'frame': frame_data['frame_id']
                })
            
            # Détecter hors-jeu
            offsides = self.detect_offside(
                frame_data['team_1'],
                frame_data['team_2'],
                frame_data['ball'],
                attacking_team=1  # À déterminer dynamiquement
            )
            
            for offside in offsides:
                events.append({
                    'type': 'offside',
                    'player_id': offside['player_id'],
                    'frame': frame_data['frame_id']
                })
            
            # Détecter corner
            corner = self.detect_corner(frame_data['ball'])
            if corner:
                events.append({
                    **corner,
                    'frame': frame_data['frame_id']
                })
            
            # Enregistrer position ballon
            self.ball_history.append(frame_data['ball'])
        
        self.events.extend(events)
        return events
    
    def get_event_summary(self) -> Dict:
        """
        Retourne un résumé de tous les événements détectés
        """
        summary = defaultdict(int)
        
        for event in self.events:
            summary[event['type']] += 1
        
        return dict(summary)


if __name__ == "__main__":
    # Test du détecteur d'événements
    detector = EventDetector()
    print("✅ Détecteur d'événements initialisé")
    
    # Test détection de possession
    ball_pos = (500, 400)
    players = [
        {'id': 1, 'bbox': [490, 390, 510, 410]},
        {'id': 2, 'bbox': [300, 300, 320, 320]}
    ]
    
    possessor = detector.detect_ball_possession(ball_pos, players)
    print(f"Joueur en possession: {possessor['id'] if possessor else 'Aucun'}")
