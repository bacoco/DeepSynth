#!/usr/bin/env python3
"""
🚀 Lanceur pour créer des datasets séparés par langue
Crée: deepsynth-fr, deepsynth-es, deepsynth-de, deepsynth-en-news, etc.

Usage:
    python run_separate_datasets.py
"""

import os
import sys
from pathlib import Path
from separate_datasets_builder import main as run_separate_builder

def check_environment():
    """Check if environment is properly configured."""
    print("🔧 Vérification de l'environnement...")

    # Check HF_TOKEN
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        print("❌ HF_TOKEN introuvable!")
        print("💡 Veuillez définir votre token HuggingFace:")
        print("   export HF_TOKEN=your_token_here")
        print("   Ou l'ajouter à votre fichier .env")
        return False

    print("✅ Token HuggingFace trouvé")

    # Check .env file
    env_file = Path('.env')
    if env_file.exists():
        print("✅ Fichier .env trouvé")
    else:
        print("⚠️  Fichier .env non trouvé (optionnel)")

    return True

def show_datasets_info():
    """Show information about datasets to be created."""
    print("\\n🎯 DATASETS SÉPARÉS À CRÉER (ORDRE DE PRIORITÉ)")
    print("=" * 60)
    print("📊 Datasets de sortie:")
    print("  🥇 deepsynth-en-news   - CNN/DailyMail (~287k exemples) [PRIORITÉ 1]")
    print("  🥈 deepsynth-en-arxiv  - arXiv Scientific (~50k exemples) [PRIORITÉ 2]")
    print("  🥉 deepsynth-en-xsum   - BBC XSum (~50k exemples) [PRIORITÉ 3]")
    print("  🇫🇷 deepsynth-fr        - MLSUM Français (~392k exemples) [PRIORITÉ 4]")
    print("  🇪🇸 deepsynth-es        - MLSUM Espagnol (~266k exemples) [PRIORITÉ 5]")
    print("  🇩🇪 deepsynth-de        - MLSUM Allemand (~220k exemples) [PRIORITÉ 6]")
    print("  📜 deepsynth-en-legal   - BillSum Legal (~22k exemples) [PRIORITÉ 7]")
    print("  " + "─" * 50)
    print("  📊 TOTAL: 7 datasets séparés")

    print("\\n🚀 Avantages des datasets séparés:")
    print("  ✅ Un dataset par langue/domaine")
    print("  ✅ Téléchargement sélectif possible")
    print("  ✅ Entraînement spécialisé par langue")
    print("  ✅ Gestion plus facile des versions")
    print("  ✅ Partage ciblé avec la communauté")

    arxiv_limit = int(os.getenv('ARXIV_IMAGE_SAMPLES', '50000'))
    print(f"\\n⚙️  Configuration arXiv: {arxiv_limit:,} échantillons max")
    print("⏱️  Temps estimé: 6-12 heures (selon le matériel)")
    print("💾 Espace disque nécessaire: ~15GB temporaire")

def main():
    """Run the separate datasets pipeline."""
    print("🌍 DEEPSEEK DATASETS SÉPARÉS PAR LANGUE")
    print("=" * 70)

    # Check environment
    if not check_environment():
        sys.exit(1)

    # Show datasets info
    show_datasets_info()

    # Confirm execution
    print("\\n❓ Prêt à créer les 7 datasets séparés?")
    print("   Cela va traiter ~1.29M échantillons et les uploader sur HuggingFace")

    try:
        response = input("Continuer? [y/N]: ").strip().lower()
        if response not in ['y', 'yes', 'o', 'oui']:
            print("⏹️  Pipeline annulé par l'utilisateur")
            sys.exit(0)
    except KeyboardInterrupt:
        print("\\n⏹️  Pipeline annulé par l'utilisateur")
        sys.exit(0)

    print("\\n🚀 Démarrage du pipeline de datasets séparés...")
    print("💡 Vous pouvez interrompre avec Ctrl+C et reprendre plus tard")

    try:
        # Run the separate datasets builder
        run_separate_builder()

        print("\\n🎉 TOUS LES DATASETS CRÉÉS AVEC SUCCÈS!")
        print("🔗 Vos datasets sont maintenant disponibles sur HuggingFace")

    except KeyboardInterrupt:
        print("\\n⏸️  Pipeline interrompu par l'utilisateur")
        print("💡 Progrès sauvegardé - relancez pour reprendre")
        sys.exit(0)

    except Exception as e:
        print(f"\\n❌ Échec du pipeline: {e}")
        print("💡 Vérifiez l'erreur ci-dessus et relancez pour reprendre")
        sys.exit(1)

if __name__ == "__main__":
    main()
