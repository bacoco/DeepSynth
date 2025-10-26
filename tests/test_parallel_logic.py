#!/usr/bin/env python3
"""
Test de la logique de parallélisation
Teste avec un petit nombre d'échantillons pour vérifier que tout fonctionne
"""

import os
import sys
import time
import logging

from deepsynth.pipelines.parallel_processing.parallel_datasets_builder import (
    ParallelDatasetsBuilder,
)

# Configuration du logging pour le test
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_parallel_logic():
    """Test de base de la logique parallèle avec limite de 500 échantillons"""
    print("🧪 TEST DE LA LOGIQUE DE PARALLÉLISATION")
    print("="*50)

    # Créer le builder avec moins de workers pour le test
    builder = ParallelDatasetsBuilder(max_workers=2)

    # Modifier la configuration pour limiter à 500 échantillons par dataset pour le test
    for dataset_config in builder.datasets_config:
        dataset_config['max_samples'] = 500  # Limite stricte pour les tests

    # Tester avec seulement 2 datasets pour commencer
    test_datasets = ['CNN/DailyMail', 'XSum BBC']

    print(f"📋 Test avec {len(test_datasets)} datasets (max 500 échantillons chacun):")
    for dataset in test_datasets:
        print(f"  - {dataset}")

    print("\n🚀 Démarrage du test parallèle...")
    start_time = time.time()

    try:
        # Lancer le traitement parallèle
        results = builder.run_parallel_processing(
            selected_datasets=test_datasets
        )

        duration = time.time() - start_time

        print(f"\n✅ Test terminé en {duration:.1f} secondes")
        print(f"📊 Résultats: {len(results)} datasets traités")

        # Analyser les résultats
        successful = [r for r in results if r['status'] in ['success', 'already_complete']]
        failed = [r for r in results if r['status'] == 'error']

        print(f"✅ Succès/Complets: {len(successful)}")
        print(f"❌ Échecs: {len(failed)}")

        if failed:
            print("\n❌ ERREURS DÉTECTÉES:")
            for result in failed:
                print(f"  - {result['dataset']}: {result.get('error', 'Erreur inconnue')}")

        return len(failed) == 0

    except Exception as e:
        print(f"\n❌ ERREUR CRITIQUE: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_dataset_independence():
    """Test que les datasets sont bien indépendants"""
    print("\n🔍 TEST D'INDÉPENDANCE DES DATASETS")
    print("="*40)

    builder = ParallelDatasetsBuilder(max_workers=1)

    # Vérifier que chaque dataset a un nom de sortie unique
    output_names = [d['output_name'] for d in builder.datasets_config]
    unique_names = set(output_names)

    if len(output_names) == len(unique_names):
        print("✅ Tous les datasets ont des noms de sortie uniques")
        return True
    else:
        print("❌ Conflit de noms de sortie détecté!")
        duplicates = [name for name in output_names if output_names.count(name) > 1]
        print(f"Doublons: {set(duplicates)}")
        return False

def main():
    """Point d'entrée principal du test"""
    print("🧪 SUITE DE TESTS POUR LA PARALLÉLISATION")
    print("="*60)

    all_passed = True

    # Test 1: Indépendance des datasets
    print("\n1️⃣ Test d'indépendance des datasets...")
    if not test_dataset_independence():
        all_passed = False

    # Test 2: Logique de parallélisation
    print("\n2️⃣ Test de la logique de parallélisation...")
    if not test_parallel_logic():
        all_passed = False

    # Résumé final
    print("\n" + "="*60)
    if all_passed:
        print("🎉 TOUS LES TESTS SONT PASSÉS!")
        print("✅ La parallélisation est prête à être utilisée")
    else:
        print("❌ CERTAINS TESTS ONT ÉCHOUÉ")
        print("🔧 Vérifiez les erreurs ci-dessus avant de continuer")

    print("="*60)
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
