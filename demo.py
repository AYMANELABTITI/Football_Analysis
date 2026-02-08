"""
Script d'exemple pour analyser une vidéo de démonstration
Génère une vidéo synthétique si aucune vidéo n'est fournie
"""

import cv2
import numpy as np
import os
from pathlib import Path
import sys

sys.path.append(os.path.dirname(__file__))

from src.utils.helpers import check_gpu_availability


def generate_demo_video(output_path: str = "data/raw/demo_match.mp4",
                       duration: int = 10, fps: int = 30):
    """
    Génère une vidéo de démonstration avec des objets en mouvement
    
    Args:
        output_path: Chemin de sortie
        duration: Durée en secondes
        fps: Images par seconde
    """
    print("🎬 Génération d'une vidéo de démonstration...")
    
    # Créer le dossier si nécessaire
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    width, height = 1280, 720
    total_frames = duration * fps
    
    # Créer le writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Positions initiales des "joueurs" et du "ballon"
    players = [
        {'pos': np.array([200.0, 360.0]), 'vel': np.array([2.0, 1.0]), 'color': (0, 0, 255)},  # Rouge
        {'pos': np.array([400.0, 360.0]), 'vel': np.array([-1.5, 1.5]), 'color': (0, 0, 255)},
        {'pos': np.array([600.0, 360.0]), 'vel': np.array([1.0, -2.0]), 'color': (255, 0, 0)},  # Bleu
        {'pos': np.array([800.0, 360.0]), 'vel': np.array([-2.0, -1.0]), 'color': (255, 0, 0)},
        {'pos': np.array([1000.0, 360.0]), 'vel': np.array([1.5, 2.0]), 'color': (255, 0, 0)},
    ]
    
    ball = {'pos': np.array([640.0, 360.0]), 'vel': np.array([3.0, 2.0])}
    
    for frame_idx in range(total_frames):
        # Créer un fond vert (terrain)
        frame = np.ones((height, width, 3), dtype=np.uint8) * np.array([34, 139, 34], dtype=np.uint8)
        
        # Dessiner les lignes du terrain
        cv2.line(frame, (0, height//2), (width, height//2), (255, 255, 255), 2)
        cv2.line(frame, (width//2, 0), (width//2, height), (255, 255, 255), 2)
        cv2.circle(frame, (width//2, height//2), 80, (255, 255, 255), 2)
        
        # Mettre à jour et dessiner les joueurs
        for player in players:
            # Mettre à jour la position
            player['pos'] += player['vel']
            
            # Rebondir sur les bords
            if player['pos'][0] <= 20 or player['pos'][0] >= width - 20:
                player['vel'][0] *= -1
            if player['pos'][1] <= 20 or player['pos'][1] >= height - 20:
                player['vel'][1] *= -1
            
            # Dessiner le joueur (rectangle)
            x, y = int(player['pos'][0]), int(player['pos'][1])
            cv2.rectangle(frame, (x-15, y-30), (x+15, y+30), player['color'], -1)
            cv2.rectangle(frame, (x-15, y-30), (x+15, y+30), (255, 255, 255), 2)
        
        # Mettre à jour et dessiner le ballon
        ball['pos'] += ball['vel']
        
        # Rebondir sur les bords
        if ball['pos'][0] <= 10 or ball['pos'][0] >= width - 10:
            ball['vel'][0] *= -1
        if ball['pos'][1] <= 10 or ball['pos'][1] >= height - 10:
            ball['vel'][1] *= -1
        
        # Dessiner le ballon
        bx, by = int(ball['pos'][0]), int(ball['pos'][1])
        cv2.circle(frame, (bx, by), 10, (0, 255, 255), -1)
        cv2.circle(frame, (bx, by), 12, (0, 0, 0), 2)
        
        # Ajouter du texte
        cv2.putText(frame, f"Demo Match - Frame {frame_idx+1}/{total_frames}",
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Écrire la frame
        out.write(frame)
    
    out.release()
    print(f"✅ Vidéo de démonstration créée: {output_path}")
    print(f"   Durée: {duration}s, FPS: {fps}, Résolution: {width}x{height}")


def run_demo_analysis():
    """
    Exécute une analyse complète sur la vidéo de démonstration
    """
    print("\n" + "="*60)
    print("🏟️  DÉMONSTRATION DU SYSTÈME D'ANALYSE DE FOOTBALL")
    print("="*60)
    
    # Vérifier le GPU
    check_gpu_availability()
    
    # Vérifier si une vidéo de démo existe
    demo_video = "data/raw/demo_match.mp4"
    
    if not os.path.exists(demo_video):
        print("\n📹 Aucune vidéo de démo trouvée")
        generate_demo_video(demo_video, duration=10, fps=30)
    else:
        print(f"\n✅ Vidéo de démo trouvée: {demo_video}")
    
    # Importer et exécuter le pipeline
    print("\n🚀 Lancement de l'analyse...")
    
    from main import FootballAnalysisPipeline
    
    pipeline = FootballAnalysisPipeline()
    pipeline.process_video(
        demo_video,
        output_path="results/videos/demo_analyzed.mp4",
        show_preview=False,
        save_stats=True
    )
    
    print("\n" + "="*60)
    print("✅ DÉMONSTRATION TERMINÉE!")
    print("="*60)
    print("\n📁 Résultats disponibles dans:")
    print("   - results/videos/demo_analyzed.mp4 (vidéo annotée)")
    print("   - results/stats/demo_match_*.csv (statistiques)")
    print("   - results/stats/demo_match_report.png (rapport visuel)")
    print("   - results/stats/demo_match_summary.txt (résumé)")


def run_performance_benchmark():
    """
    Exécute un benchmark de performance
    """
    print("\n" + "="*60)
    print("⚡ BENCHMARK DE PERFORMANCE")
    print("="*60)
    
    from src.utils.helpers import benchmark_gpu
    import time
    
    # Vérifier GPU
    check_gpu_availability()
    
    # Benchmark GPU si disponible
    import torch
    if torch.cuda.is_available():
        benchmark_gpu()
    
    # Test de détection
    print("\n🧪 Test de détection YOLOv8...")
    from src.detection.detector import FootballDetector
    
    detector = FootballDetector()
    
    # Créer une image de test
    test_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    
    # Mesurer le temps
    iterations = 10
    start = time.time()
    
    for i in range(iterations):
        players, balls = detector.detect_frame(test_frame)
    
    elapsed = time.time() - start
    fps = iterations / elapsed
    
    print(f"✅ FPS de détection: {fps:.2f}")
    print(f"   Temps moyen par frame: {elapsed/iterations*1000:.2f}ms")


def interactive_demo():
    """
    Mode interactif pour explorer les fonctionnalités
    """
    print("\n" + "="*60)
    print("🎮 MODE INTERACTIF")
    print("="*60)
    
    while True:
        print("\nQue voulez-vous faire?")
        print("1. Générer une vidéo de démonstration")
        print("2. Analyser la vidéo de démonstration")
        print("3. Exécuter un benchmark de performance")
        print("4. Tester les composants individuels")
        print("5. Quitter")
        
        choice = input("\nVotre choix (1-5): ").strip()
        
        if choice == '1':
            duration = input("Durée (secondes, défaut=10): ").strip()
            duration = int(duration) if duration else 10
            generate_demo_video(duration=duration)
            
        elif choice == '2':
            run_demo_analysis()
            
        elif choice == '3':
            run_performance_benchmark()
            
        elif choice == '4':
            print("\n🧪 Test des composants...")
            os.system("python test_system.py")
            
        elif choice == '5':
            print("\n👋 Au revoir!")
            break
        
        else:
            print("❌ Choix invalide")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Script de démonstration")
    parser.add_argument('--generate', action='store_true',
                       help='Générer une vidéo de démonstration')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyser la vidéo de démonstration')
    parser.add_argument('--benchmark', action='store_true',
                       help='Exécuter un benchmark')
    parser.add_argument('--interactive', action='store_true',
                       help='Mode interactif')
    parser.add_argument('--duration', type=int, default=10,
                       help='Durée de la vidéo de démo (secondes)')
    
    args = parser.parse_args()
    
    if args.generate:
        generate_demo_video(duration=args.duration)
    elif args.analyze:
        run_demo_analysis()
    elif args.benchmark:
        run_performance_benchmark()
    elif args.interactive:
        interactive_demo()
    else:
        # Par défaut: mode interactif
        interactive_demo()
