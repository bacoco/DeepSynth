#!/usr/bin/env python3
"""
Script de lancement pour le traitement parallèle des datasets
"""

import os
import sys

from deepsynth.data.transforms.text_to_image import DEEPSEEK_OCR_RESOLUTIONS

from .parallel_datasets_builder import ParallelDatasetsPipeline

def run_parallel_datasets_cli():
    """Point d'entrée principal"""
    print("🚀 TRAITEMENT PARALLÈLE DES DATASETS DEEPSYNTH")
    print("="*60)
    print("Ce script traite plusieurs datasets en parallèle pour accélérer le processus.")
    print("Chaque dataset est indépendant et sera uploadé sur HuggingFace séparément.")
    print()
    
    # Vérifier le token HuggingFace
    if not os.getenv('HF_TOKEN'):
        print("❌ Variable d'environnement HF_TOKEN manquante!")
        print("   Ajoutez votre token HuggingFace dans le fichier .env")
        return 1
    
    # Configuration
    print("⚙️ CONFIGURATION")
    print("-" * 30)
    
    max_workers = input("Nombre de processus parallèles (défaut: 3, max recommandé: 4): ").strip()
    try:
        max_workers = int(max_workers) if max_workers else 3
        if max_workers < 1:
            max_workers = 1
        elif max_workers > 6:
            print("⚠️ Attention: Plus de 6 processus peut surcharger le système")
            max_workers = 6
    except ValueError:
        max_workers = 3
    
    print(f"✅ Utilisation de {max_workers} processus parallèles")

    # Multi-resolution: Always enabled with all sizes
    print("\n🔍 MULTI-RÉSOLUTION (DeepSeek OCR)")
    print("-" * 30)
    print("✅ Multi-résolution activée: TOUTES les résolutions seront générées")
    sizes_list = list(DEEPSEEK_OCR_RESOLUTIONS.items())
    for idx, (name, size) in enumerate(sizes_list, start=1):
        print(f"  {idx}. {name:<6} ({size[0]}×{size[1]})")

    multi_resolution = True
    resolution_sizes = None  # All sizes

    # Options de traitement
    print("\n📋 OPTIONS DE TRAITEMENT")
    print("-" * 30)
    print("1. Traiter tous les datasets (recommandé)")
    print("2. Traiter des datasets spécifiques")
    print("3. Mode test rapide (500 échantillons par dataset)")

    choice = input("\nVotre choix (1-3): ").strip()

    # Créer le pipeline
    pipeline = ParallelDatasetsPipeline(
        max_workers=max_workers,
        multi_resolution=multi_resolution,
        resolution_sizes=resolution_sizes
    )
    
    print(f"\n📊 DATASETS DISPONIBLES ({len(pipeline.datasets_config)} au total)")
    print("-" * 50)
    for i, dataset in enumerate(pipeline.datasets_config, 1):
        max_samples = dataset.get('max_samples')
        samples_info = f" (max {max_samples:,})" if max_samples else " (complet)"
        print(f"{i:2d}. {dataset['name']:<20} → {dataset['output_name']}{samples_info}")
    
    try:
        if choice == "1":
            # Tous les datasets
            print(f"\n🚀 DÉMARRAGE DU TRAITEMENT COMPLET")
            print("=" * 50)
            print("⚠️  Ceci peut prendre plusieurs heures selon la taille des datasets")
            
            confirm = input("Continuer? (o/N): ").strip().lower()
            if confirm not in ['o', 'oui', 'y', 'yes']:
                print("❌ Annulé par l'utilisateur")
                return 0
            
            results = pipeline.run_parallel_processing()
            
        elif choice == "2":
            # Datasets spécifiques
            print(f"\n📝 SÉLECTION DE DATASETS")
            print("-" * 30)
            selected_indices = input("Indices des datasets à traiter (ex: 1,2,3): ").strip()
            
            try:
                indices = [int(x.strip()) - 1 for x in selected_indices.split(",")]
                selected_datasets = []
                
                for i in indices:
                    if 0 <= i < len(pipeline.datasets_config):
                        selected_datasets.append(pipeline.datasets_config[i]['name'])
                    else:
                        print(f"⚠️ Index {i+1} invalide, ignoré")
                
                if not selected_datasets:
                    print("❌ Aucun dataset valide sélectionné")
                    return 1
                
                print(f"\n✅ Datasets sélectionnés: {', '.join(selected_datasets)}")
                results = pipeline.run_parallel_processing(selected_datasets=selected_datasets)
                
            except ValueError:
                print("❌ Format invalide")
                return 1
                
        elif choice == "3":
            # Mode test
            print(f"\n🧪 MODE TEST RAPIDE")
            print("-" * 30)
            print("Traitement de 500 échantillons maximum par dataset pour test")
            
            # Limiter tous les datasets à 500 échantillons
            for dataset_config in pipeline.datasets_config:
                dataset_config['max_samples'] = 500
            
            # Demander quels datasets tester
            test_choice = input("Tester tous les datasets (o) ou seulement quelques-uns (n)? (o/N): ").strip().lower()
            
            if test_choice in ['o', 'oui', 'y', 'yes']:
                results = pipeline.run_parallel_processing()
            else:
                selected_indices = input("Indices des datasets à tester (ex: 1,3): ").strip()
                try:
                    indices = [int(x.strip()) - 1 for x in selected_indices.split(",")]
                    selected_datasets = [
                        pipeline.datasets_config[i]['name']
                        for i in indices
                        if 0 <= i < len(pipeline.datasets_config)
                    ]
                    
                    if selected_datasets:
                        results = pipeline.run_parallel_processing(selected_datasets=selected_datasets)
                    else:
                        print("❌ Aucun dataset valide sélectionné")
                        return 1
                except ValueError:
                    print("❌ Format invalide")
                    return 1
        else:
            print("❌ Choix invalide")
            return 1
        
        # Résumé final
        successful = [r for r in results if r['status'] == 'success']
        failed = [r for r in results if r['status'] == 'error']
        
        print(f"\n🎯 RÉSUMÉ FINAL")
        print("=" * 40)
        print(f"✅ Réussis: {len(successful)}")
        print(f"❌ Échoués: {len(failed)}")
        
        if successful:
            print(f"\n🔗 DATASETS CRÉÉS:")
            for result in successful:
                if 'repo_name' in result:
                    print(f"  - https://huggingface.co/datasets/{result['repo_name']}")
        
        if failed:
            print(f"\n❌ ÉCHECS À INVESTIGUER:")
            for result in failed:
                print(f"  - {result['dataset']}: {result.get('error', 'Erreur inconnue')}")
        
        print(f"\n📋 Consultez 'parallel_datasets.log' pour les détails complets.")
        
        return 0 if len(failed) == 0 else 1
        
    except KeyboardInterrupt:
        print(f"\n⏸️ INTERRUPTION DÉTECTÉE")
        print("Le traitement peut être relancé - il reprendra où il s'est arrêté grâce à la logique incrémentale.")
        return 0
    except Exception as e:
        print(f"\n❌ ERREUR CRITIQUE: {str(e)}")
        return 1

if __name__ == "__main__":
    exit_code = run_parallel_datasets_cli()
    sys.exit(exit_code)