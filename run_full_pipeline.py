#!/usr/bin/env python3
"""
Script de production pour traiter TOUS les datasets en parallèle
Crée les 7 datasets séparés sur HuggingFace

Supports multi-resolution image generation for DeepSeek OCR training.
"""
import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

from deepsynth.data.transforms.text_to_image import DEEPSEEK_OCR_RESOLUTIONS

# Charger .env
env_path = Path(__file__).parent / '.env'
load_dotenv(env_path)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process all datasets with multi-resolution image generation (always enabled)"
    )
    parser.add_argument(
        '--single-resolution',
        action='store_true',
        help='Generate ONLY single resolution images (disables multi-resolution)'
    )
    parser.add_argument(
        '--resolution-sizes',
        nargs='+',
        choices=list(DEEPSEEK_OCR_RESOLUTIONS.keys()),
        help='Specific resolution sizes to generate (default: all)'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=7,
        help='Maximum number of parallel workers (default: 7)'
    )
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()

    # Vérifier le token
    if not os.getenv('HF_TOKEN'):
        print("❌ HF_TOKEN non trouvé dans .env")
        sys.exit(1)

    print("🌍 TRAITEMENT COMPLET DES 7 DATASETS MULTILINGUES")
    print("=" * 70)
    print(f"✅ HF_TOKEN configuré")
    print(f"✅ HF_USERNAME: {os.getenv('HF_USERNAME')}")
    print(f"✅ ARXIV_IMAGE_SAMPLES: {os.getenv('ARXIV_IMAGE_SAMPLES', '50000')}")

    # Importer le pipeline
    from deepsynth.pipelines.parallel import ParallelDatasetsPipeline

    # Configuration pour production COMPLÈTE
    print("\n📊 CONFIGURATION PRODUCTION")
    print("-" * 70)
    print(f"⚙️  Nombre de workers: {args.max_workers} (parallélisme)")
    print("📦 Mode: Production complète (tous les échantillons disponibles)")
    print("🔄 Reprise automatique si dataset existant détecté")
    print("📤 Upload automatique tous les 5000 échantillons")

    # Multi-resolution info (always enabled unless --single-resolution flag)
    multi_resolution = not args.single_resolution

    if multi_resolution:
        if args.resolution_sizes:
            sizes_str = ', '.join(args.resolution_sizes)
            print(f"🔍 Multi-résolution activée: {sizes_str}")
        else:
            all_sizes = "/".join(DEEPSEEK_OCR_RESOLUTIONS.keys())
            print(f"🔍 Multi-résolution activée: toutes les tailles ({all_sizes})")
    else:
        print("📸 Mode résolution unique (image standard)")

    pipeline = ParallelDatasetsPipeline(
        max_workers=args.max_workers,
        multi_resolution=multi_resolution,
        resolution_sizes=args.resolution_sizes
    )

    # Afficher les datasets à traiter
    print(f"\n📋 DATASETS À TRAITER ({len(pipeline.datasets_config)} au total)")
    print("-" * 70)
    total_estimated = 0
    for i, dataset in enumerate(pipeline.datasets_config, 1):
        max_samples = dataset.get('max_samples')
        if max_samples:
            samples_info = f"max {max_samples:,} échantillons"
            total_estimated += max_samples
        else:
            samples_info = "complet (non limité)"

        priority_emoji = "🥇" if dataset['priority'] == 1 else "🥈" if dataset['priority'] == 2 else "🥉" if dataset['priority'] == 3 else "📊"
        print(f"{priority_emoji} {i}. {dataset['name']:<20} → {dataset['output_name']:<25} ({samples_info})")

    print(f"\n📊 Total estimé: ~{total_estimated:,}+ échantillons à traiter")
    print("⏱️  Temps estimé: 6-12 heures (selon matériel et connexion)")
    print("💾 Espace disque requis: ~15GB temporaire")

    print("\n⚠️  NOTES IMPORTANTES:")
    print("  • Le traitement peut être interrompu (Ctrl+C) et reprendra automatiquement")
    print("  • Les datasets déjà créés seront mis à jour incrémentalement")
    print("  • Les logs détaillés sont dans 'parallel_datasets.log'")
    print("  • Chaque dataset sera visible sur HuggingFace dès son upload")

    print("\n" + "=" * 70)

    try:
        print("🚀 DÉMARRAGE DU TRAITEMENT PARALLÈLE COMPLET")
        print("=" * 70)

        # Lancer le traitement de TOUS les datasets
        results = pipeline.run_parallel_processing()

        # Résumé final
        successful = [r for r in results if r['status'] == 'success']
        failed = [r for r in results if r['status'] == 'error']

        print("\n" + "=" * 70)
        print("🎉 TRAITEMENT COMPLET TERMINÉ!")
        print("=" * 70)
        print(f"✅ Réussis: {len(successful)}/{len(results)}")
        print(f"❌ Échoués: {len(failed)}/{len(results)}")

        if successful:
            print(f"\n🔗 DATASETS CRÉÉS SUR HUGGINGFACE:")
            for result in successful:
                if 'repo_name' in result:
                    print(f"  📦 https://huggingface.co/datasets/{result['repo_name']}")

        if failed:
            print(f"\n❌ DATASETS EN ÉCHEC:")
            for result in failed:
                print(f"  ⚠️  {result['dataset']}: {result.get('error', 'Erreur inconnue')}")

        print(f"\n📋 Consultez 'parallel_datasets.log' pour les détails complets")

        return 0 if len(failed) == 0 else 1

    except KeyboardInterrupt:
        print("\n⏸️  TRAITEMENT INTERROMPU PAR L'UTILISATEUR")
        print("💡 Relancez ce script pour reprendre où vous vous êtes arrêté")
        print("   Les datasets partiellement traités seront complétés automatiquement")
        return 0

    except Exception as e:
        print(f"\n❌ ERREUR CRITIQUE: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 Vérifiez 'parallel_datasets.log' pour plus de détails")
        return 1

if __name__ == '__main__':
    sys.exit(main())
