#!/usr/bin/env python3
"""
Parallel Dataset Builder - Traite plusieurs datasets en parallèle
Chaque dataset est indépendant et peut être traité simultanément
"""

import os
import sys
import time
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
import threading
from datetime import datetime
from ..separate_datasets_builder import SeparateDatasetBuilder

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('parallel_datasets.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ParallelDatasetsBuilder:
    def __init__(self, max_workers=3):
        """
        Initialise le builder parallèle

        Args:
            max_workers: Nombre maximum de processus parallèles
        """
        self.max_workers = max_workers
        # Get arXiv limit from environment
        try:
            arxiv_limit = int(os.getenv('ARXIV_IMAGE_SAMPLES', '50000'))
        except ValueError:
            arxiv_limit = 50000
        if arxiv_limit <= 0:
            arxiv_limit = 10000

        self.datasets_config = [
            {
                'name': 'CNN/DailyMail',
                'output_name': 'deepsynth-en-news',
                'priority': 1,
                'dataset_name': 'cnn_dailymail',
                'dataset_config': '3.0.0',
                'text_column': 'article',
                'summary_column': 'highlights',
                'max_samples': None
            },
            {
                'name': 'arXiv',
                'output_name': 'deepsynth-en-arxiv',
                'priority': 2,
                'dataset_name': 'ccdv/arxiv-summarization',
                'dataset_config': None,
                'text_column': 'article',
                'summary_column': 'abstract',
                'max_samples': arxiv_limit
            },
            {
                'name': 'XSum BBC',
                'output_name': 'deepsynth-en-xsum',
                'priority': 3,
                'dataset_name': 'Rexhaif/xsum_reduced',
                'dataset_config': None,
                'text_column': 'text',
                'summary_column': 'target',
                'max_samples': None
            },
            {
                'name': 'MLSUM Français',
                'output_name': 'deepsynth-fr',
                'priority': 4,
                'dataset_name': 'MLSUM',
                'dataset_config': 'fr',
                'text_column': 'text',
                'summary_column': 'summary',
                'max_samples': None
            },
            {
                'name': 'MLSUM Espagnol',
                'output_name': 'deepsynth-es',
                'priority': 5,
                'dataset_name': 'MLSUM',
                'dataset_config': 'es',
                'text_column': 'text',
                'summary_column': 'summary',
                'max_samples': None
            },
            {
                'name': 'MLSUM Allemand',
                'output_name': 'deepsynth-de',
                'priority': 6,
                'dataset_name': 'MLSUM',
                'dataset_config': 'de',
                'text_column': 'text',
                'summary_column': 'summary',
                'max_samples': None
            },
            {
                'name': 'BillSum Legal',
                'output_name': 'deepsynth-en-legal',
                'priority': 7,
                'dataset_name': 'billsum',
                'dataset_config': None,
                'text_column': 'text',
                'summary_column': 'summary',
                'max_samples': None
            }
        ]

        # Trier par priorité
        self.datasets_config.sort(key=lambda x: x['priority'])

    def process_single_dataset(self, dataset_config, shared_stats):
        """
        Traite un seul dataset (fonction pour processus parallèle)

        Args:
            dataset_config: Configuration du dataset
            shared_stats: Dictionnaire partagé pour les statistiques

        Returns:
            dict: Résultats du traitement
        """
        try:
            logger.info(f"🚀 Démarrage du traitement: {dataset_config['name']}")

            # Importer les modules nécessaires dans le processus
            from huggingface_hub import login, whoami
            import os

            # Login HuggingFace
            login(token=os.getenv('HF_TOKEN'))
            username = whoami()['name']

            # Créer le builder avec un work_dir unique pour ce dataset
            work_dir = f"./work_separate_{dataset_config['output_name']}"
            builder = SeparateDatasetBuilder(work_dir=work_dir)

            # Traitement du dataset
            start_time = time.time()

            # Vérifier l'état existant sur HuggingFace
            repo_name = f"{username}/{dataset_config['output_name']}"
            progress_info = builder.check_existing_dataset_progress(repo_name)

            if progress_info['exists']:
                existing_count = progress_info['total_processed']
                logger.info(f"📊 {dataset_config['name']}: {existing_count} échantillons déjà traités")
            else:
                existing_count = 0
                logger.info(f"📊 {dataset_config['name']}: Nouveau dataset à créer")

            # Traiter le dataset avec la méthode existante
            builder.process_and_upload_dataset(
                name=dataset_config['dataset_name'],
                subset=dataset_config['dataset_config'],
                text_field=dataset_config['text_column'],
                summary_field=dataset_config['summary_column'],
                username=username,
                max_samples=dataset_config.get('max_samples')
            )

            duration = time.time() - start_time

            # Mettre à jour les statistiques partagées
            shared_stats['completed'] = shared_stats.get('completed', 0) + 1
            shared_stats['total_duration'] = shared_stats.get('total_duration', 0) + duration

            logger.info(f"✅ {dataset_config['name']} terminé en {duration/60:.1f} minutes")

            return {
                'dataset': dataset_config['name'],
                'status': 'success',
                'duration': duration,
                'existing_count': existing_count,
                'repo_name': repo_name
            }

        except Exception as e:
            logger.error(f"❌ Erreur lors du traitement de {dataset_config['name']}: {str(e)}")
            return {
                'dataset': dataset_config['name'],
                'status': 'error',
                'error': str(e),
                'duration': time.time() - start_time if 'start_time' in locals() else 0
            }

    def monitor_progress(self, shared_stats, total_datasets):
        """
        Monitore et affiche la progression globale
        """
        start_time = time.time()

        while shared_stats['completed'] < total_datasets:
            elapsed = time.time() - start_time
            completed = shared_stats['completed']

            if completed > 0:
                avg_time = elapsed / completed
                eta = avg_time * (total_datasets - completed)

                logger.info(f"📈 Progression globale: {completed}/{total_datasets} datasets terminés")
                logger.info(f"⏱️  Temps écoulé: {elapsed/60:.1f}min, ETA: {eta/60:.1f}min")
                logger.info(f"📊 Total échantillons traités: {shared_stats['total_processed']}")

            time.sleep(30)  # Mise à jour toutes les 30 secondes

    def run_parallel_processing(self, selected_datasets=None, test_mode=False):
        """
        Lance le traitement parallèle de tous les datasets

        Args:
            selected_datasets: Liste des noms de datasets à traiter (None = tous)
            test_mode: Si True, traite seulement quelques échantillons pour test
        """
        logger.info("🚀 Démarrage du traitement parallèle des datasets")

        # Filtrer les datasets si spécifié
        datasets_to_process = self.datasets_config
        if selected_datasets:
            datasets_to_process = [
                d for d in self.datasets_config
                if d['name'] in selected_datasets or d['output_name'] in selected_datasets
            ]

        logger.info(f"📋 Datasets à traiter: {len(datasets_to_process)}")
        for dataset in datasets_to_process:
            logger.info(f"  - {dataset['name']} (priorité {dataset['priority']})")

        # Statistiques partagées entre processus
        with Manager() as manager:
            shared_stats = manager.dict({
                'completed': 0,
                'total_processed': 0,
                'total_duration': 0
            })

            # Démarrer le monitoring en arrière-plan
            monitor_thread = threading.Thread(
                target=self.monitor_progress,
                args=(shared_stats, len(datasets_to_process))
            )
            monitor_thread.daemon = True
            monitor_thread.start()

            # Traitement parallèle
            start_time = time.time()
            results = []

            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # Soumettre tous les jobs
                future_to_dataset = {
                    executor.submit(self.process_single_dataset, dataset_config, shared_stats): dataset_config
                    for dataset_config in datasets_to_process
                }

                # Collecter les résultats au fur et à mesure
                for future in as_completed(future_to_dataset):
                    dataset_config = future_to_dataset[future]
                    try:
                        result = future.result()
                        results.append(result)

                        if result['status'] == 'success':
                            logger.info(f"✅ {result['dataset']} terminé avec succès")
                        elif result['status'] == 'already_complete':
                            logger.info(f"✅ {result['dataset']} déjà complet")
                        else:
                            logger.error(f"❌ {result['dataset']} a échoué: {result.get('error', 'Erreur inconnue')}")

                    except Exception as e:
                        logger.error(f"❌ Erreur lors de la récupération du résultat pour {dataset_config['name']}: {str(e)}")
                        results.append({
                            'dataset': dataset_config['name'],
                            'status': 'error',
                            'error': str(e)
                        })

            total_duration = time.time() - start_time

            # Arrêter le monitoring
            shared_stats['completed'] = len(datasets_to_process)

            # Afficher le résumé final
            self.print_final_summary(results, total_duration)

            return results

    def print_final_summary(self, results, total_duration):
        """Affiche le résumé final des résultats"""
        logger.info("\n" + "="*80)
        logger.info("📊 RÉSUMÉ FINAL DU TRAITEMENT PARALLÈLE")
        logger.info("="*80)

        successful = [r for r in results if r['status'] == 'success']
        already_complete = [r for r in results if r['status'] == 'already_complete']
        failed = [r for r in results if r['status'] == 'error']

        logger.info(f"✅ Succès: {len(successful)}")
        logger.info(f"✅ Déjà complets: {len(already_complete)}")
        logger.info(f"❌ Échecs: {len(failed)}")
        logger.info(f"⏱️  Durée totale: {total_duration/60:.1f} minutes")

        if successful:
            logger.info("\n🎉 DATASETS TRAITÉS AVEC SUCCÈS:")
            for result in successful:
                duration = result.get('duration', 0)
                processed = result.get('result', {}).get('processed_count', 0)
                logger.info(f"  - {result['dataset']}: {processed} échantillons en {duration/60:.1f}min")

        if already_complete:
            logger.info("\n✅ DATASETS DÉJÀ COMPLETS:")
            for result in already_complete:
                logger.info(f"  - {result['dataset']}: {result['existing_count']}/{result['total_count']} ({result['progress']:.1f}%)")

        if failed:
            logger.info("\n❌ DATASETS EN ÉCHEC:")
            for result in failed:
                logger.info(f"  - {result['dataset']}: {result.get('error', 'Erreur inconnue')}")

        logger.info("="*80)

def main():
    """Point d'entrée principal"""
    print("🚀 Parallel Dataset Builder")
    print("="*50)

    # Configuration
    max_workers = int(input("Nombre de processus parallèles (défaut: 3): ") or "3")

    print("\nOptions de traitement:")
    print("1. Traiter tous les datasets")
    print("2. Traiter des datasets spécifiques")
    print("3. Mode test (quelques échantillons)")

    choice = input("\nVotre choix (1-3): ").strip()

    builder = ParallelDatasetsBuilder(max_workers=max_workers)

    if choice == "1":
        # Tous les datasets
        results = builder.run_parallel_processing()

    elif choice == "2":
        # Datasets spécifiques
        print("\nDatasets disponibles:")
        for i, dataset in enumerate(builder.datasets_config, 1):
            print(f"{i}. {dataset['name']} ({dataset['output_name']})")

        selected_indices = input("\nIndices des datasets à traiter (ex: 1,2,3): ").strip()
        try:
            indices = [int(x.strip()) - 1 for x in selected_indices.split(",")]
            selected_datasets = [builder.datasets_config[i]['name'] for i in indices if 0 <= i < len(builder.datasets_config)]

            if selected_datasets:
                results = builder.run_parallel_processing(selected_datasets=selected_datasets)
            else:
                print("❌ Aucun dataset valide sélectionné")
                return
        except ValueError:
            print("❌ Format invalide")
            return

    elif choice == "3":
        # Mode test
        print("🧪 Mode test activé - traitement de quelques échantillons seulement")
        results = builder.run_parallel_processing(test_mode=True)

    else:
        print("❌ Choix invalide")
        return

    print(f"\n✅ Traitement terminé! Consultez 'parallel_datasets.log' pour les détails.")

if __name__ == "__main__":
    main()
