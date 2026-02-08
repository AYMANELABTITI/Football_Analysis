"""
Pipeline professionnel FIFA-style
Sortie propre avec skeleton tracking et multi-fenêtres
"""

import cv2
import sys
import os
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import yaml
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from detection.detector import ProfessionalFootballDetector
from tracking.tracker import FootballTracker
from visualization.visualizer import ProfessionalVisualizer


class ProfessionalPipeline:
    """Pipeline de qualité professionnelle FIFA"""
    
    def __init__(self, config_path: str = "config/config_pro.yaml",
                 calibration_p1: Optional[Tuple[float, float]] = None,
                 calibration_p2: Optional[Tuple[float, float]] = None,
                 calibration_distance_m: Optional[float] = None):
        """Initialise le pipeline professionnel"""
        print("="*70)
        print(" "*15 + "🏟️  PROFESSIONAL FOOTBALL ANALYSIS")
        print(" "*20 + "FIFA-Style Output")
        print("="*70)

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f) or {}

        detection_cfg = self.config.get('detection', {})
        event_cfg = self.config.get('events', {})
        field_cfg = self.config.get('field', {})
        self.ball_hold_frames = int(detection_cfg.get('ball_hold_frames', 6))
        self.ball_link_distance_px = float(detection_cfg.get('ball_link_distance_px', 120))
        self.ball_control_distance_px = float(event_cfg.get('ball_control_distance_px', 80))
        self.pass_min_distance_px = float(event_cfg.get('pass_min_distance_px', 80))
        self.pass_min_duration_frames = int(event_cfg.get('pass_min_duration_frames', 2))
        self.pass_max_duration_frames = int(event_cfg.get('pass_max_duration_frames', 90))
        self.pass_min_separation_frames = int(event_cfg.get('pass_min_separation_frames', 2))
        self.pass_max_angle_deg = float(event_cfg.get('pass_max_angle_deg', 75))
        self.pass_min_ball_displacement_px = float(event_cfg.get('pass_min_ball_displacement_px', 18))
        self.pass_min_owner_frames = int(event_cfg.get('pass_min_owner_frames', 2))

        # Modules
        self.detector = ProfessionalFootballDetector(config_path)
        self.tracker_team1 = FootballTracker(config_path)
        self.tracker_team2 = FootballTracker(config_path)
        self.visualizer = ProfessionalVisualizer()
        
        # Historiques pour heatmap
        self.positions_team1 = []
        self.positions_team2 = []
        
        # Etats ball / possession
        self.last_ball_center = None
        self.last_ball_bbox = None
        self.ball_missing_streak = 0
        self.current_owner = None
        self.last_pass_frame = -10**9
        self.ball_position_history = []
        
        # Donnees export
        self.frame_records = []
        self.pass_events = []
        self.player_metrics = {}
        self.distance_scale_px_to_m = float(field_cfg.get('distance_scale_px_to_m', 0.0))
        cfg_p1 = self._parse_point(field_cfg.get('calibration_p1'))
        cfg_p2 = self._parse_point(field_cfg.get('calibration_p2'))
        cfg_distance_m = float(field_cfg.get('calibration_distance_m', 68.0))
        self.calibration_p1 = calibration_p1 if calibration_p1 is not None else cfg_p1
        self.calibration_p2 = calibration_p2 if calibration_p2 is not None else cfg_p2
        if calibration_distance_m is not None and calibration_distance_m > 0:
            self.calibration_distance_m = float(calibration_distance_m)
        else:
            self.calibration_distance_m = cfg_distance_m
        
        print("\n✅ Pipeline professionnel prêt!\n")
    
    def process_video_professional(self, video_path: str, output_path: str = None,
                                  show_skeleton: bool = True, 
                                  create_heatmap: bool = True,
                                  create_minimap: bool = True):
        """
        Traite la vidéo avec sortie professionnelle
        
        Args:
            video_path: Vidéo d'entrée
            output_path: Vidéo de sortie
            show_skeleton: Afficher les skeletons
            create_heatmap: Créer la heatmap
            create_minimap: Créer la mini-map
        """
        print(f"🎬 Traitement: {video_path}\n")
        
        # Ouvrir la vidéo
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Erreur: Impossible d'ouvrir {video_path}")
            return
        
        # Propriétés
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📊 Résolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}\n")
        
        # Sortie
        if output_path is None:
            output_path = f"results/videos/professional_{Path(video_path).stem}.mp4"
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Writer avec codec H.264 pour meilleure qualité
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Traitement
        frame_idx = 0
        start_time = time.time()
        
        print("🔄 Traitement en cours...")
        print("-"*70)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_start = time.time()
            
            # 1. DÉTECTION AVANCÉE avec poses
            players, balls, _ = self.detector.detect_with_pose(frame)
            
            # 2. CLASSIFICATION D'ÉQUIPES
            teams = self.detector.classify_teams_advanced(frame, players)
            
            team_1 = teams['team_1']
            team_2 = teams['team_2']
            
            # 3. TRACKING
            # Convertir au format attendu par le tracker
            team1_for_tracking = [[p['bbox'][0], p['bbox'][1], p['bbox'][2], p['bbox'][3], p['conf'], 0] 
                                  for p in team_1]
            team2_for_tracking = [[p['bbox'][0], p['bbox'][1], p['bbox'][2], p['bbox'][3], p['conf'], 0]
                                  for p in team_2]
            
            team1_tracks = self.tracker_team1.update(team1_for_tracking)
            team2_tracks = self.tracker_team2.update(team2_for_tracking)
            self._update_player_metrics(team1_tracks, 'team_1', fps, frame_idx)
            self._update_player_metrics(team2_tracks, 'team_2', fps, frame_idx)
            
            # Associer les données détaillées aux tracks
            team1_with_details = self._merge_tracks_with_detections(team1_tracks, team_1)
            team2_with_details = self._merge_tracks_with_detections(team2_tracks, team_2)
            
            # 4. BALLON
            ball = self._select_best_ball(balls)
            ball_pos = None
            ball_interpolated = False
            
            if ball is not None:
                ball_pos = ((ball['bbox'][0] + ball['bbox'][2]) / 2,
                           (ball['bbox'][1] + ball['bbox'][3]) / 2)
                self.last_ball_center = ball_pos
                self.last_ball_bbox = ball['bbox']
                self.ball_missing_streak = 0
            else:
                ball = self._recover_ball_from_memory()
                if ball is not None:
                    ball_pos = ((ball['bbox'][0] + ball['bbox'][2]) / 2,
                               (ball['bbox'][1] + ball['bbox'][3]) / 2)
                    ball_interpolated = True
            
            owner, pass_event = self._update_possession_and_passes(
                frame_idx, ball_pos, team1_with_details, team2_with_details, fps
            )
            self._append_ball_history(frame_idx, ball_pos)
            
            # 5. VISUALISATION PROFESSIONNELLE
            # HUD en haut
            current_fps = 1.0 / (time.time() - frame_start) if frame_idx > 0 else 0
            stats = {
                'team_1_count': len(team1_with_details),
                'team_2_count': len(team2_with_details),
                'frame': frame_idx + 1,
                'total_frames': total_frames,
                'fps': current_fps,
                'events': len(self.pass_events)
            }
            frame = self.visualizer.draw_professional_hud(frame, stats)
            
            # Dessiner les joueurs avec skeletons
            for player in team1_with_details:
                frame = self.visualizer.draw_clean_player(
                    frame, player,
                    self.visualizer.colors['team_1'],
                    player.get('id'),
                    show_skeleton=show_skeleton
                )
                # Enregistrer position pour heatmap
                self.positions_team1.append(player['center'])
            
            for player in team2_with_details:
                frame = self.visualizer.draw_clean_player(
                    frame, player,
                    self.visualizer.colors['team_2'],
                    player.get('id'),
                    show_skeleton=show_skeleton
                )
                self.positions_team2.append(player['center'])
            
            # Dessiner le ballon
            if ball is not None:
                frame = self.visualizer.draw_clean_ball(frame, ball)
            
            # Mini-map dans le coin (optionnel)
            if create_minimap and frame_idx % 2 == 0:  # Tous les 2 frames
                minimap = self.visualizer.create_mini_map(
                    team1_with_details, team2_with_details, ball_pos
                )
                # Incruster la mini-map en bas à droite
                mm_h, mm_w = minimap.shape[:2]
                frame[height-mm_h-20:height-20, width-mm_w-20:width-20] = minimap
            
            self.frame_records.append({
                'frame': frame_idx + 1,
                'time_s': round(frame_idx / fps, 3) if fps > 0 else 0.0,
                'team_1_players': len(team1_with_details),
                'team_2_players': len(team2_with_details),
                'ball_detected': int(ball is not None and not ball_interpolated),
                'ball_interpolated': int(ball_interpolated),
                'ball_x': round(ball_pos[0], 2) if ball_pos is not None else None,
                'ball_y': round(ball_pos[1], 2) if ball_pos is not None else None,
                'owner_team': owner['team'] if owner else None,
                'owner_player_id': owner['player_id'] if owner else None,
                'pass_event': pass_event['pass_id'] if pass_event else None
            })
            
            # Sauvegarder
            out.write(frame)
            
            # Progress
            if frame_idx % 30 == 0:
                progress = (frame_idx / total_frames) * 100
                elapsed = time.time() - start_time
                eta = (elapsed / (frame_idx + 1)) * (total_frames - frame_idx - 1)
                print(f"\r[{progress:5.1f}%] Frame {frame_idx+1}/{total_frames} | "
                      f"FPS: {current_fps:4.1f} | ETA: {eta:.0f}s", end='')
            
            frame_idx += 1
        
        # Cleanup
        cap.release()
        out.release()
        
        total_time = time.time() - start_time
        avg_fps = total_frames / total_time
        
        print(f"\n\n{'='*70}")
        print(f"✅ Traitement terminé!")
        print(f"   Temps total: {total_time:.1f}s")
        print(f"   FPS moyen: {avg_fps:.1f}")
        print(f"   Vidéo sauvegardée: {output_path}")
        
        # Créer la heatmap finale
        if create_heatmap and self.positions_team1 and self.positions_team2:
            print(f"\n🎨 Génération de la heatmap...")
            heatmap_path = output_path.replace('.mp4', '_heatmap.png')
            heatmap = self.visualizer.create_clean_heatmap({
                'team_1': self.positions_team1,
                'team_2': self.positions_team2
            }, (width, height))
            cv2.imwrite(heatmap_path, heatmap)
            print(f"   Heatmap: {heatmap_path}")
        
        self._export_analysis_files(output_path, fps, total_frames, total_time, (width, height))
        
        print(f"{'='*70}\n")
    
    @staticmethod
    def _parse_point(value) -> Optional[Tuple[float, float]]:
        """Parse un point [x, y] depuis config ou tuple."""
        if value is None:
            return None
        if isinstance(value, (list, tuple)) and len(value) == 2:
            try:
                return float(value[0]), float(value[1])
            except (TypeError, ValueError):
                return None
        return None
    
    def _resolve_distance_scale(self, frame_size: Tuple[int, int]) -> float:
        """Resout la conversion px -> m via config/calibration/fallback."""
        width = max(int(frame_size[0]), 1)
        
        if self.distance_scale_px_to_m > 0:
            return self.distance_scale_px_to_m
        
        if (
            self.calibration_p1 is not None and
            self.calibration_p2 is not None and
            self.calibration_distance_m is not None and
            self.calibration_distance_m > 0
        ):
            px_distance = float(np.linalg.norm(np.array(self.calibration_p2) - np.array(self.calibration_p1)))
            if px_distance > 1e-6:
                return float(self.calibration_distance_m / px_distance)
        
        return float(105.0 / width)
    
    @staticmethod
    def _vector_angle_deg(v1: np.ndarray, v2: np.ndarray) -> Optional[float]:
        """Retourne l'angle (deg) entre 2 vecteurs ou None si indefini."""
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 <= 1e-6 or n2 <= 1e-6:
            return None
        cosine = float(np.dot(v1, v2) / (n1 * n2))
        cosine = max(-1.0, min(1.0, cosine))
        return float(np.degrees(np.arccos(cosine)))
    
    def _append_ball_history(self, frame_idx: int, ball_pos: Optional[Tuple[float, float]]) -> None:
        """Stocke l'historique court des positions ballon."""
        if ball_pos is None:
            return
        self.ball_position_history.append((frame_idx, (float(ball_pos[0]), float(ball_pos[1]))))
        if len(self.ball_position_history) > 300:
            self.ball_position_history = self.ball_position_history[-300:]
    
    def _get_recent_ball_position(self, frame_idx: int, max_age_frames: int = 8) -> Optional[Tuple[float, float]]:
        """Recupere la derniere position ballon suffisamment recente."""
        for fidx, pos in reversed(self.ball_position_history):
            if frame_idx - fidx <= max_age_frames:
                return pos
        return None
    
    def _update_player_metrics(self, tracks: list, team_name: str, fps: int, frame_idx: int) -> None:
        """Met a jour trajectoires et distances pour chaque joueur tracke."""
        for track in tracks:
            player_id = int(track.get('id', -1))
            if player_id < 0:
                continue
            
            key = f"{team_name}_{player_id}"
            positions = [(float(p[0]), float(p[1])) for p in track.get('positions', [])]
            distance_px = float(track.get('distance', 0.0))
            speed_px_s = 0.0
            if len(positions) >= 2 and fps > 0:
                x1, y1 = positions[-2]
                x2, y2 = positions[-1]
                speed_px_s = (((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5) * fps
            
            record = self.player_metrics.get(key, {
                'team': team_name,
                'player_id': player_id,
                'positions': [],
                'distance_px': 0.0,
                'max_speed_px_s': 0.0,
                'frames_seen': 0,
                'last_seen_frame': 0
            })
            if positions:
                record['positions'] = positions
            record['distance_px'] = max(record['distance_px'], distance_px)
            record['max_speed_px_s'] = max(record['max_speed_px_s'], speed_px_s)
            record['frames_seen'] = max(record['frames_seen'], int(track.get('hits', 0)))
            record['last_seen_frame'] = frame_idx + 1
            self.player_metrics[key] = record
    
    def _build_player_distance_tables(self, frame_size: Tuple[int, int]):
        """Construit les tableaux de distance joueur et equipe."""
        px_to_meter = self._resolve_distance_scale(frame_size)
        
        player_rows = []
        for _, metric in self.player_metrics.items():
            positions = metric.get('positions', [])
            if len(positions) < 2:
                continue
            distance_px = float(metric.get('distance_px', 0.0))
            player_rows.append({
                'team': metric['team'],
                'player_id': int(metric['player_id']),
                'distance_px': round(distance_px, 2),
                'distance_m_est': round(distance_px * px_to_meter, 2),
                'max_speed_px_s': round(float(metric.get('max_speed_px_s', 0.0)), 2),
                'frames_seen': int(metric.get('frames_seen', 0)),
                'trajectory_points': len(positions),
                'last_seen_frame': int(metric.get('last_seen_frame', 0))
            })
        
        player_df = pd.DataFrame(player_rows)
        if player_df.empty:
            player_df = pd.DataFrame(columns=[
                'team', 'player_id', 'distance_px', 'distance_m_est', 'max_speed_px_s',
                'frames_seen', 'trajectory_points', 'last_seen_frame'
            ])
        
        team_df = (
            player_df.groupby('team', as_index=False)
            .agg(
                total_distance_px=('distance_px', 'sum'),
                total_distance_m_est=('distance_m_est', 'sum'),
                avg_distance_px=('distance_px', 'mean'),
                players_count=('player_id', 'nunique')
            )
            if not player_df.empty else
            pd.DataFrame(columns=[
                'team', 'total_distance_px', 'total_distance_m_est', 'avg_distance_px', 'players_count'
            ])
        )
        
        if not player_df.empty:
            player_df = player_df.sort_values(by='distance_px', ascending=False).reset_index(drop=True)
        
        return player_df, team_df, px_to_meter
    
    def _save_trajectory_map(self, output_file: Path, frame_size: Tuple[int, int]) -> Path:
        """Sauvegarde une carte des trajectoires joueurs."""
        width, height = frame_size
        traj_img = np.full((height, width, 3), (34, 139, 34), dtype=np.uint8)
        traj_img = self.visualizer._draw_field_lines(traj_img)
        
        # Garder les joueurs les plus actifs pour lisibilite
        metrics = [
            m for m in self.player_metrics.values()
            if len(m.get('positions', [])) >= 2
        ]
        metrics = sorted(metrics, key=lambda m: m.get('distance_px', 0.0), reverse=True)[:22]
        
        for metric in metrics:
            team = metric['team']
            color = self.visualizer.colors.get(team, (255, 255, 255))
            pts = np.array(
                [(int(p[0]), int(p[1])) for p in metric.get('positions', [])],
                dtype=np.int32
            )
            if len(pts) < 2:
                continue
            cv2.polylines(traj_img, [pts], False, color, 2)
            end_pt = tuple(pts[-1])
            cv2.circle(traj_img, end_pt, 4, color, -1)
        
        trajectory_path = output_file.with_name(f"{output_file.stem}_trajectories.png")
        cv2.imwrite(str(trajectory_path), traj_img)
        return trajectory_path
    
    def _save_distance_chart(self, output_file: Path, player_df: pd.DataFrame) -> Optional[Path]:
        """Sauvegarde un graphique des distances par joueur."""
        if player_df.empty:
            return None
        
        top_df = player_df.head(16).copy()
        labels = [f"{'T1' if t == 'team_1' else 'T2'}-{pid}" for t, pid in zip(top_df['team'], top_df['player_id'])]
        colors = ['#4da3ff' if t == 'team_2' else '#ff9a4d' for t in top_df['team']]
        
        plt.figure(figsize=(13, 7))
        plt.bar(labels, top_df['distance_px'], color=colors)
        plt.title("Distance parcourue par joueur (px) - Top 16")
        plt.xlabel("Joueur")
        plt.ylabel("Distance (px)")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        chart_path = output_file.with_name(f"{output_file.stem}_distance_chart.png")
        plt.savefig(chart_path, dpi=160)
        plt.close()
        return chart_path
    
    def _save_match_dashboard(self, output_file: Path, frame_df: pd.DataFrame, pass_df: pd.DataFrame) -> Optional[Path]:
        """Sauvegarde un dashboard statistique global du match."""
        if frame_df.empty:
            return None
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        
        axes[0].plot(frame_df['frame'], frame_df['team_1_players'], label='Team 1 players', color='#ff9a4d')
        axes[0].plot(frame_df['frame'], frame_df['team_2_players'], label='Team 2 players', color='#4da3ff')
        axes[0].set_ylabel("Players detected")
        axes[0].set_title("Match tracking statistics")
        axes[0].legend(loc='upper right')
        axes[0].grid(alpha=0.25)
        
        ball_status = frame_df['ball_detected'] + frame_df['ball_interpolated']
        axes[1].plot(frame_df['frame'], ball_status, label='Ball tracked (1/0)', color='#2ca02c')
        if not pass_df.empty:
            cumulative = np.zeros(len(frame_df), dtype=np.float32)
            for end_frame in pass_df['end_frame']:
                if 1 <= int(end_frame) <= len(cumulative):
                    cumulative[int(end_frame) - 1:] += 1
            axes[1].plot(frame_df['frame'], cumulative, label='Cumulative passes', color='#d62728')
        axes[1].set_xlabel("Frame")
        axes[1].set_ylabel("Events")
        axes[1].legend(loc='upper left')
        axes[1].grid(alpha=0.25)
        
        plt.tight_layout()
        dashboard_path = output_file.with_name(f"{output_file.stem}_stats_dashboard.png")
        plt.savefig(dashboard_path, dpi=160)
        plt.close(fig)
        return dashboard_path
    
    def _select_best_ball(self, balls: list) -> Optional[Dict]:
        """Selectionne la meilleure detection de ballon."""
        if not balls:
            return None

        if self.last_ball_center is None:
            return max(balls, key=lambda b: b.get('conf', 0.0))

        best_ball = None
        best_score = -10**9
        for candidate in balls:
            bbox = candidate.get('bbox')
            if not bbox:
                continue
            center_x = (bbox[0] + bbox[2]) / 2.0
            center_y = (bbox[1] + bbox[3]) / 2.0
            distance = ((center_x - self.last_ball_center[0]) ** 2 +
                        (center_y - self.last_ball_center[1]) ** 2) ** 0.5
            score = float(candidate.get('conf', 0.0)) - (distance / max(self.ball_link_distance_px, 1.0))
            if score > best_score:
                best_score = score
                best_ball = candidate

        if best_ball is None:
            return None

        bbox = best_ball['bbox']
        center_x = (bbox[0] + bbox[2]) / 2.0
        center_y = (bbox[1] + bbox[3]) / 2.0
        distance = ((center_x - self.last_ball_center[0]) ** 2 +
                    (center_y - self.last_ball_center[1]) ** 2) ** 0.5
        if distance > self.ball_link_distance_px * 2.5 and best_ball.get('conf', 0.0) < 0.55:
            return None

        return best_ball

    def _recover_ball_from_memory(self) -> Optional[Dict]:
        """Recupere une position ballon estimee sur quelques frames."""
        self.ball_missing_streak += 1

        if self.last_ball_center is None:
            return None
        if self.ball_missing_streak > self.ball_hold_frames:
            return None

        cx, cy = self.last_ball_center
        if self.last_ball_bbox is not None:
            bw = max(10.0, self.last_ball_bbox[2] - self.last_ball_bbox[0])
            bh = max(10.0, self.last_ball_bbox[3] - self.last_ball_bbox[1])
        else:
            bw, bh = 16.0, 16.0

        return {
            'bbox': [cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0],
            'conf': 0.0,
            'class': 32,
            'interpolated': True
        }

    def _detect_ball_owner(self, ball_pos: Optional[Tuple[float, float]],
                           team1_players: list, team2_players: list) -> Optional[Dict]:
        """Trouve le joueur le plus proche du ballon."""
        if ball_pos is None:
            return None

        candidates = []
        for team_name, players in (('team_1', team1_players), ('team_2', team2_players)):
            for player in players:
                center = player.get('center')
                if center is None:
                    continue
                distance = ((center[0] - ball_pos[0]) ** 2 + (center[1] - ball_pos[1]) ** 2) ** 0.5
                if distance <= self.ball_control_distance_px:
                    candidates.append({
                        'team': team_name,
                        'player_id': player.get('id'),
                        'center': center,
                        'distance_px': distance
                    })

        if not candidates:
            return None

        return min(candidates, key=lambda c: c['distance_px'])

    def _update_possession_and_passes(self, frame_idx: int,
                                      ball_pos: Optional[Tuple[float, float]],
                                      team1_players: list,
                                      team2_players: list,
                                      fps: int) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Met a jour la possession et detecte des passes simples."""
        owner = self._detect_ball_owner(ball_pos, team1_players, team2_players)
        pass_event = None

        if owner is None:
            return None, None

        if self.current_owner is None:
            self.current_owner = {
                'team': owner['team'],
                'player_id': owner.get('player_id'),
                'start_frame': frame_idx,
                'last_center': owner['center'],
                'start_ball_pos': ball_pos,
                'last_ball_pos': ball_pos
            }
            return owner, None

        same_owner = (
            owner['team'] == self.current_owner['team'] and
            owner.get('player_id') == self.current_owner.get('player_id')
        )
        if same_owner:
            self.current_owner['last_center'] = owner['center']
            if ball_pos is not None:
                self.current_owner['last_ball_pos'] = ball_pos
            return owner, None

        previous_owner = self.current_owner
        valid_ids = (
            previous_owner.get('player_id') is not None and
            owner.get('player_id') is not None and
            previous_owner.get('player_id') != owner.get('player_id')
        )
        same_team = owner['team'] == previous_owner['team']

        if same_team and valid_ids:
            start_pos = previous_owner['last_center']
            end_pos = owner['center']
            pass_distance = ((end_pos[0] - start_pos[0]) ** 2 + (end_pos[1] - start_pos[1]) ** 2) ** 0.5
            duration_frames = frame_idx - previous_owner['start_frame']
            owner_frames = max(0, duration_frames)
            
            ball_start = previous_owner.get('start_ball_pos') or previous_owner.get('last_ball_pos') or start_pos
            ball_end = ball_pos or self._get_recent_ball_position(frame_idx, max_age_frames=8) or end_pos
            ball_vec = np.array([ball_end[0] - ball_start[0], ball_end[1] - ball_start[1]], dtype=np.float32)
            pass_vec = np.array([end_pos[0] - start_pos[0], end_pos[1] - start_pos[1]], dtype=np.float32)
            ball_displacement = float(np.linalg.norm(ball_vec))
            pass_angle_deg = self._vector_angle_deg(pass_vec, ball_vec)
            angle_valid = pass_angle_deg is not None and pass_angle_deg <= self.pass_max_angle_deg

            if (
                pass_distance >= self.pass_min_distance_px and
                self.pass_min_duration_frames <= duration_frames <= self.pass_max_duration_frames and
                owner_frames >= self.pass_min_owner_frames and
                ball_displacement >= self.pass_min_ball_displacement_px and
                angle_valid and
                (frame_idx - self.last_pass_frame) >= self.pass_min_separation_frames
            ):
                pass_id = len(self.pass_events) + 1
                pass_event = {
                    'pass_id': pass_id,
                    'team': owner['team'],
                    'from_player': int(previous_owner['player_id']),
                    'to_player': int(owner['player_id']),
                    'start_frame': int(previous_owner['start_frame']) + 1,
                    'end_frame': frame_idx + 1,
                    'start_time_s': round(previous_owner['start_frame'] / fps, 3) if fps > 0 else 0.0,
                    'end_time_s': round(frame_idx / fps, 3) if fps > 0 else 0.0,
                    'duration_frames': int(duration_frames),
                    'distance_px': round(pass_distance, 2),
                    'ball_displacement_px': round(ball_displacement, 2),
                    'pass_angle_deg': round(pass_angle_deg, 2) if pass_angle_deg is not None else None,
                    'start_x': round(start_pos[0], 2),
                    'start_y': round(start_pos[1], 2),
                    'end_x': round(end_pos[0], 2),
                    'end_y': round(end_pos[1], 2)
                }
                self.pass_events.append(pass_event)
                self.last_pass_frame = frame_idx

        self.current_owner = {
            'team': owner['team'],
            'player_id': owner.get('player_id'),
            'start_frame': frame_idx,
            'last_center': owner['center'],
            'start_ball_pos': ball_pos,
            'last_ball_pos': ball_pos
        }

        return owner, pass_event

    def _export_analysis_files(self, output_path: str, fps: int, total_frames: int,
                               total_time: float, frame_size: Tuple[int, int]) -> None:
        """Exporte les statistiques en XLSX et la carte graphique des passes."""
        output_file = Path(output_path)
        report_xlsx = output_file.with_name(f"{output_file.stem}_analysis.xlsx")
        pass_map_path = output_file.with_name(f"{output_file.stem}_passes.png")

        frame_df = pd.DataFrame(self.frame_records)
        pass_df = pd.DataFrame(self.pass_events)
        if pass_df.empty:
            pass_df = pd.DataFrame(columns=[
                'pass_id', 'team', 'from_player', 'to_player', 'start_frame', 'end_frame',
                'start_time_s', 'end_time_s', 'duration_frames', 'distance_px',
                'ball_displacement_px', 'pass_angle_deg',
                'start_x', 'start_y', 'end_x', 'end_y'
            ])

        ball_detected_frames = int(frame_df['ball_detected'].sum()) if not frame_df.empty else 0
        ball_interpolated_frames = int(frame_df['ball_interpolated'].sum()) if not frame_df.empty else 0

        pass_network_df = (
            pass_df.groupby(['team', 'from_player', 'to_player'])
            .size()
            .reset_index(name='pass_count')
            if not pass_df.empty else pd.DataFrame(columns=['team', 'from_player', 'to_player', 'pass_count'])
        )
        player_df, team_distance_df, px_to_meter = self._build_player_distance_tables(frame_size)

        summary_df = pd.DataFrame([{
            'total_frames': total_frames,
            'processed_time_s': round(total_time, 2),
            'avg_fps': round(total_frames / max(total_time, 1e-9), 2),
            'passes_total': int(len(self.pass_events)),
            'passes_team_1': int((pass_df['team'] == 'team_1').sum()) if not pass_df.empty else 0,
            'passes_team_2': int((pass_df['team'] == 'team_2').sum()) if not pass_df.empty else 0,
            'ball_detected_frames': ball_detected_frames,
            'ball_interpolated_frames': ball_interpolated_frames,
            'ball_tracking_coverage_pct': round(
                100.0 * (ball_detected_frames + ball_interpolated_frames) / max(total_frames, 1), 2
            ),
            'tracked_players': int(player_df['player_id'].nunique()) if not player_df.empty else 0,
            'total_player_distance_px': round(float(player_df['distance_px'].sum()), 2) if not player_df.empty else 0.0,
            'distance_scale_px_to_m': round(px_to_meter, 5)
        }])

        try:
            with pd.ExcelWriter(report_xlsx, engine='openpyxl') as writer:
                summary_df.to_excel(writer, sheet_name='summary', index=False)
                frame_df.to_excel(writer, sheet_name='frame_details', index=False)
                pass_df.to_excel(writer, sheet_name='passes', index=False)
                pass_network_df.to_excel(writer, sheet_name='pass_network', index=False)
                player_df.to_excel(writer, sheet_name='player_distances', index=False)
                team_distance_df.to_excel(writer, sheet_name='team_distances', index=False)
            print(f"   XLSX report: {report_xlsx}")
        except Exception as exc:
            fallback_base = output_file.with_name(f"{output_file.stem}_analysis")
            summary_df.to_csv(f"{fallback_base}_summary.csv", index=False)
            frame_df.to_csv(f"{fallback_base}_frame_details.csv", index=False)
            pass_df.to_csv(f"{fallback_base}_passes.csv", index=False)
            pass_network_df.to_csv(f"{fallback_base}_pass_network.csv", index=False)
            player_df.to_csv(f"{fallback_base}_player_distances.csv", index=False)
            team_distance_df.to_csv(f"{fallback_base}_team_distances.csv", index=False)
            print(f"   XLSX export failed ({exc}). CSV files exported instead.")

        pass_map = self.visualizer.create_pass_map(self.pass_events, frame_size)
        cv2.imwrite(str(pass_map_path), pass_map)
        print(f"   Pass map: {pass_map_path}")
        
        trajectory_path = self._save_trajectory_map(output_file, frame_size)
        print(f"   Trajectories map: {trajectory_path}")
        
        distance_chart_path = self._save_distance_chart(output_file, player_df)
        if distance_chart_path is not None:
            print(f"   Distance chart: {distance_chart_path}")
        
        dashboard_path = self._save_match_dashboard(output_file, frame_df, pass_df)
        if dashboard_path is not None:
            print(f"   Match dashboard: {dashboard_path}")

    def _merge_tracks_with_detections(self, tracks: list, detections: list) -> list:
        """Associe les informations de tracking aux détections"""
        result = []
        
        for track in tracks:
            track_bbox = track['bbox']
            
            # Trouver la détection correspondante
            best_match = None
            best_iou = 0
            
            for det in detections:
                det_bbox = det['bbox']
                iou = self._calculate_iou(track_bbox, det_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_match = det
            
            # Fusionner
            if best_match and best_iou > 0.3:
                merged = {
                    'id': track['id'],
                    'bbox': track_bbox,
                    'conf': best_match['conf'],
                    'center': best_match['center'],
                    'keypoints': best_match.get('keypoints')
                }
                result.append(merged)
        
        return result
    
    def _calculate_iou(self, box1, box2):
        """Calcule IoU"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        xi_min = max(x1_min, x2_min)
        yi_min = max(y1_min, y2_min)
        xi_max = min(x1_max, x2_max)
        yi_max = min(y1_max, y2_max)
        
        if xi_max <= xi_min or yi_max <= yi_min:
            return 0.0
        
        intersection = (xi_max - xi_min) * (yi_max - yi_min)
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0


def main():
    """Point d'entree professionnel"""
    import argparse

    def parse_point_arg(value: Optional[str]) -> Optional[Tuple[float, float]]:
        if value is None:
            return None
        parts = [p.strip() for p in value.split(',')]
        if len(parts) != 2:
            raise ValueError(f"Point invalide: {value}. Format attendu: x,y")
        return float(parts[0]), float(parts[1])

    parser = argparse.ArgumentParser(description="Professional Football Analysis")
    parser.add_argument('video', type=str, help='Video du match')
    parser.add_argument('--output', type=str, default=None, help='Fichier de sortie')
    parser.add_argument('--no-skeleton', action='store_true', help='Desactiver skeleton tracking')
    parser.add_argument('--no-heatmap', action='store_true', help='Desactiver heatmap')
    parser.add_argument('--no-minimap', action='store_true', help='Desactiver mini-map')
    parser.add_argument('--config', type=str, default='config/config_pro.yaml', help='Config')
    parser.add_argument('--calib-p1', type=str, default=None, help='Point calibration 1 (x,y)')
    parser.add_argument('--calib-p2', type=str, default=None, help='Point calibration 2 (x,y)')
    parser.add_argument('--calib-distance-m', type=float, default=68.0, help='Distance reelle entre p1 et p2 (m)')

    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Fichier non trouve: {args.video}")
        return

    try:
        calib_p1 = parse_point_arg(args.calib_p1)
        calib_p2 = parse_point_arg(args.calib_p2)
    except ValueError as exc:
        print(str(exc))
        return

    if (calib_p1 is None) != (calib_p2 is None):
        print("Merci de fournir --calib-p1 et --calib-p2 ensemble, ou aucun des deux.")
        return

    pipeline = ProfessionalPipeline(
        args.config,
        calibration_p1=calib_p1,
        calibration_p2=calib_p2,
        calibration_distance_m=args.calib_distance_m
    )
    pipeline.process_video_professional(
        args.video,
        output_path=args.output,
        show_skeleton=not args.no_skeleton,
        create_heatmap=not args.no_heatmap,
        create_minimap=not args.no_minimap
    )


if __name__ == "__main__":
    main()
