#!/usr/bin/env python3
"""
Test de la logique incrémentale avec un petit échantillon
"""

import os
import sys
from pathlib import Path
from huggingface_hub import login, whoami

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from deepsynth.pipelines import SeparateDatasetsPipeline

def test_incremental_logic():
    """Test avec un petit échantillon pour vérifier la logique"""
    print("🧪 TEST DE LA LOGIQUE INCRÉMENTALE")
    print("=" * 50)
    
    # Login
    login(token=os.getenv('HF_TOKEN'))
    username = whoami()['name']
    
    builder = SeparateDatasetsPipeline()
    
    # Test avec CNN/DailyMail - seulement 10 échantillons pour le test
    print("\\n🔬 Test avec CNN/DailyMail (10 échantillons)")
    
    try:
        builder.process_and_upload_dataset(
            name='cnn_dailymail',
            subset='3.0.0', 
            text_field='article',
            summary_field='highlights',
            username=username,
            max_samples=10  # Très petit pour le test
        )
        
        print("\\n✅ Test réussi!")
        print("\\n🔄 Maintenant, relancez le même test pour voir la logique de reprise...")
        
    except Exception as e:
        print(f"\\n❌ Erreur pendant le test: {e}")
        raise

if __name__ == "__main__":
    test_incremental_logic()