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

from deepsynth.data.transforms.text_to_image import DEEPSEEK_OCR_RESOLUTIONS
from deepsynth.pipelines._dataset_processor import OptimizedDatasetPipeline

__all__ = ["ParallelDatasetsPipeline"]

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

class ParallelDatasetsPipeline:
    def __init__(self, max_workers=7, multi_resolution=True, resolution_sizes=None):
        """
        Initialise le builder parallèle

        Args:
            max_workers: Nombre maximum de processus parallèles
            multi_resolution: If True, generate multiple resolution images (default: True)
            resolution_sizes: List of resolution names to generate.
                             Options: ['tiny', 'small', 'base', 'large', 'gundam']
                             None means all sizes. Only used if multi_resolution=True.
        """
        self.max_workers = max_workers  # 7 = one per dataset for full parallelism
        self.multi_resolution = multi_resolution
        if self.multi_resolution:
            if resolution_sizes:
                filtered = []
                for size in resolution_sizes:
                    if size in DEEPSEEK_OCR_RESOLUTIONS and size not in filtered:
                        filtered.append(size)
                self.resolution_sizes = filtered or list(DEEPSEEK_OCR_RESOLUTIONS.keys())
            else:
                self.resolution_sizes = list(DEEPSEEK_OCR_RESOLUTIONS.keys())
        else:
            self.resolution_sizes = None
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

    def process_single_dataset(self, dataset_config, shared_stats, username):
        """
        Traite un seul dataset (fonction pour processus parallèle)

        Args:
            dataset_config: Configuration du dataset
            shared_stats: Dictionnaire partagé pour les statistiques
            username: HuggingFace username (évite rate limit sur whoami())

        Returns:
            dict: Résultats du traitement
        """
        try:
            logger.info(f"🚀 Démarrage du traitement: {dataset_config['name']}")

            # Importer les modules nécessaires dans le processus
            from huggingface_hub import login
            import os
            import time

            # Login HuggingFace (pas de whoami() ici pour éviter rate limit)
            # Small delay to avoid rate limiting on login
            time.sleep(dataset_config.get('priority', 1) * 0.5)  # Stagger logins
            login(token=os.getenv('HF_TOKEN'))

            # Créer le builder avec un work_dir unique pour ce dataset
            # auto_upload=True: Upload immediately when batch is ready
            work_dir = f"./work_separate_{dataset_config['output_name']}"

            # Get multi-resolution settings from dataset_config (passed from run_parallel_processing)
            multi_resolution = dataset_config.get('multi_resolution', False)
            resolution_sizes = dataset_config.get('resolution_sizes', None)

            builder = OptimizedDatasetPipeline(
                work_dir=work_dir,
                auto_upload=True,
                multi_resolution=multi_resolution,
                resolution_sizes=resolution_sizes
            )

            # Traitement du dataset
            start_time = time.time()

            # Vérifier l'état existant sur HuggingFace (métadonnées seulement)
            repo_name = f"{username}/{dataset_config['output_name']}"
            processed_keys = builder.check_processed_indices(repo_name)

            if processed_keys:
                existing_count = len(processed_keys)
                logger.info(f"📊 {dataset_config['name']}: {existing_count} échantillons déjà traités")
            else:
                existing_count = 0
                logger.info(f"📊 {dataset_config['name']}: Nouveau dataset à créer")

            # Traiter le dataset avec la nouvelle méthode optimisée
            builder.process_and_batch_dataset(
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
        # Login ONCE before starting workers to avoid rate limiting
        from huggingface_hub import login, whoami
        import os

        login(token=os.getenv('HF_TOKEN'))
        username = whoami()['name']
        logger.info(f"✅ Logged in as: {username}")

        logger.info("🚀 Démarrage du traitement parallèle des datasets")

        # Filtrer les datasets si spécifié
        datasets_to_process = self.datasets_config
        if selected_datasets:
            datasets_to_process = [
                d for d in self.datasets_config
                if d['name'] in selected_datasets or d['output_name'] in selected_datasets
            ]

        # Add multi-resolution settings to each dataset config
        for dataset in datasets_to_process:
            dataset['multi_resolution'] = self.multi_resolution
            dataset['resolution_sizes'] = self.resolution_sizes if self.multi_resolution else None

        logger.info(f"📋 Datasets à traiter: {len(datasets_to_process)}")
        if self.multi_resolution:
            sizes_str = ', '.join(self.resolution_sizes or [])
            logger.info(f"🔍 Multi-resolution enabled: {sizes_str}")
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
                # Soumettre tous les jobs (pass username to avoid rate limit)
                future_to_dataset = {
                    executor.submit(self.process_single_dataset, dataset_config, shared_stats, username): dataset_config
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

if __name__ == "__main__":
    from .run_parallel_datasets import run_parallel_datasets_cli

    run_parallel_datasets_cli()
