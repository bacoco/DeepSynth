#!/usr/bin/env python3
"""
Script de correction automatique des problèmes critiques identifiés dans la revue de code.
Exécuter avec: python fix_critical_issues.py
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple

# Ensure UTF-8 stdout on Windows consoles to avoid UnicodeEncodeError
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

class CriticalIssueFixer:
    """Corrige automatiquement les problèmes critiques du codebase."""

    def __init__(self):
        # Point to repository root (parent of tools/)
        self.base_path = Path(__file__).resolve().parents[1]
        self.fixes_applied = []
        self.fixes_failed = []

    def fix_all(self) -> None:
        """Applique toutes les corrections critiques."""
        print("🔧 Correction des problèmes critiques DeepSynth...\n")

        # 1. Fix NameError dans global_state.py
        self.fix_nameerror_global_state()

        # 2. Fix bare except dans incremental.py
        self.fix_bare_except_incremental()

        # 3. Fix hauteur illimitée des images
        self.fix_unbounded_image_height()

        # 4. Ajouter validation des checkpoints
        self.fix_checkpoint_validation()

        # 5. Fix double appel get_env
        self.fix_double_get_env()

        # Rapport final
        self.print_report()

    def fix_nameerror_global_state(self) -> None:
        """Corrige la variable non définie 'uploaded' dans global_state.py."""
        file_path = self.base_path / "src/deepsynth/pipelines/global_state.py"

        if not file_path.exists():
            self.fixes_failed.append(("NameError global_state.py", "Fichier non trouvé"))
            return

        try:
            content = file_path.read_text()

            # Rechercher et remplacer la ligne problématique
            old_line = "progress['total_samples'] += uploaded"
            new_line = "progress['total_samples'] += uploaded_count"

            if old_line in content:
                content = content.replace(old_line, new_line)
                file_path.write_text(content)
                self.fixes_applied.append(("NameError global_state.py:272", "Variable 'uploaded' → 'uploaded_count'"))
            else:
                self.fixes_failed.append(("NameError global_state.py", "Ligne non trouvée (peut-être déjà corrigée)"))

        except Exception as e:
            self.fixes_failed.append(("NameError global_state.py", str(e)))

    def fix_bare_except_incremental(self) -> None:
        """Remplace les 'bare except' par des exceptions spécifiques."""
        file_path = self.base_path / "src/deepsynth/pipelines/incremental.py"

        if not file_path.exists():
            self.fixes_failed.append(("Bare except incremental.py", "Fichier non trouvé"))
            return

        try:
            content = file_path.read_text()

            # Pattern pour trouver les bare except
            pattern = r'(\s+)except:\n(\s+)pass'
            replacement = r'\1except (FileNotFoundError, PermissionError):\n\2pass  # Safely ignore if repo doesn\'t exist'

            new_content, count = re.subn(pattern, replacement, content)

            if count > 0:
                file_path.write_text(new_content)
                self.fixes_applied.append(("Bare except incremental.py:300", f"Remplacé {count} 'bare except'"))
            else:
                self.fixes_failed.append(("Bare except incremental.py", "Pattern non trouvé"))

        except Exception as e:
            self.fixes_failed.append(("Bare except incremental.py", str(e)))

    def fix_unbounded_image_height(self) -> None:
        """Ajoute une limite maximale pour la hauteur des images."""
        file_path = self.base_path / "src/deepsynth/data/transforms/text_to_image.py"

        if not file_path.exists():
            self.fixes_failed.append(("Image height text_to_image.py", "Fichier non trouvé"))
            return

        try:
            lines = file_path.read_text().splitlines()

            # Chercher la ligne problématique (approximativement ligne 114)
            for i, line in enumerate(lines):
                if "total_height = max(required_height" in line or "total_height = " in line:
                    # Remplacer par une version avec limite
                    lines[i] = "        # Limit height to prevent memory explosion on very long texts"
                    lines.insert(i + 1, "        MAX_HEIGHT_MULTIPLIER = 4  # Allow up to 4x the configured height")
                    lines.insert(i + 2, "        max_allowed_height = self.max_height * MAX_HEIGHT_MULTIPLIER")
                    lines.insert(i + 3, "        total_height = min(required_height, max_allowed_height) if required_height > self.max_height else min(required_height, self.max_height)")

                    file_path.write_text('\n'.join(lines))
                    self.fixes_applied.append(("Image height text_to_image.py:114", "Ajout limite 4x hauteur max"))
                    return

            self.fixes_failed.append(("Image height text_to_image.py", "Ligne à modifier non trouvée"))

        except Exception as e:
            self.fixes_failed.append(("Image height text_to_image.py", str(e)))

    def fix_checkpoint_validation(self) -> None:
        """Ajoute la validation des checkpoints avant reprise."""
        file_path = self.base_path / "src/deepsynth/training/deepsynth_trainer_v2.py"

        if not file_path.exists():
            self.fixes_failed.append(("Checkpoint validation", "Fichier non trouvé"))
            return

        try:
            content = file_path.read_text()

            # Chercher le pattern de reprise de checkpoint
            pattern = r'(if config\.resume_from_checkpoint:)\n(\s+)(LOGGER\.info\("Resuming training from checkpoint: %s", config\.resume_from_checkpoint\))'

            replacement = r'\1\n\2# Validate checkpoint exists and is loadable\n\2checkpoint_path = Path(config.resume_from_checkpoint)\n\2if not checkpoint_path.exists():\n\2    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")\n\2if not checkpoint_path.is_dir():\n\2    raise ValueError(f"Checkpoint path is not a directory: {checkpoint_path}")\n\2\3'

            new_content, count = re.subn(pattern, replacement, content)

            # Ajouter import Path si nécessaire
            if count > 0 and "from pathlib import Path" not in new_content:
                lines = new_content.splitlines()
                # Insérer après les autres imports
                for i, line in enumerate(lines):
                    if line.startswith("import ") or line.startswith("from "):
                        continue
                    else:
                        lines.insert(i, "from pathlib import Path")
                        break
                new_content = '\n'.join(lines)

            if count > 0:
                file_path.write_text(new_content)
                self.fixes_applied.append(("Checkpoint validation trainer_v2.py", "Ajout validation existence"))
            else:
                self.fixes_failed.append(("Checkpoint validation", "Pattern non trouvé"))

        except Exception as e:
            self.fixes_failed.append(("Checkpoint validation", str(e)))

    def fix_double_get_env(self) -> None:
        """Optimise les doubles appels à get_env."""
        file_path = self.base_path / "src/deepsynth/config/env.py"

        if not file_path.exists():
            self.fixes_failed.append(("Double get_env", "Fichier non trouvé"))
            return

        try:
            lines = file_path.read_text().splitlines()

            # Chercher et corriger le pattern de double appel
            for i, line in enumerate(lines):
                if "MAX_SAMPLES_PER_SPLIT" in line and "get_env" in line and lines[i].count("get_env") > 1:
                    # Remplacer par une version optimisée
                    lines[i] = '            max_samples_per_split=self._parse_optional_int("MAX_SAMPLES_PER_SPLIT", "1000"),'

                    # Ajouter la méthode helper si elle n\'existe pas
                    method_exists = any("_parse_optional_int" in l for l in lines)
                    if not method_exists:
                        # Trouver la fin de la classe pour ajouter la méthode
                        for j in range(len(lines) - 1, 0, -1):
                            if lines[j].strip() and not lines[j].startswith(' '):
                                lines.insert(j, '    def _parse_optional_int(self, key: str, default: str) -> Optional[int]:')
                                lines.insert(j + 1, '        """Parse optional integer from environment."""')
                                lines.insert(j + 2, '        value = get_env(key, None, required=False)')
                                lines.insert(j + 3, '        return int(value) if value else None')
                                lines.insert(j + 4, '')
                                break

                    file_path.write_text('\n'.join(lines))
                    self.fixes_applied.append(("Double get_env config/env.py", "Optimisé appels multiples"))
                    return

            self.fixes_failed.append(("Double get_env", "Pattern non trouvé"))

        except Exception as e:
            self.fixes_failed.append(("Double get_env", str(e)))

    def print_report(self) -> None:
        """Affiche le rapport des corrections."""
        print("\n" + "=" * 60)
        print("📊 RAPPORT DE CORRECTIONS")
        print("=" * 60)

        if self.fixes_applied:
            print("\n✅ Corrections appliquées:")
            for issue, description in self.fixes_applied:
                print(f"  • {issue}: {description}")

        if self.fixes_failed:
            print("\n❌ Corrections échouées:")
            for issue, reason in self.fixes_failed:
                print(f"  • {issue}: {reason}")

        print("\n" + "=" * 60)
        print(f"Total: {len(self.fixes_applied)} réussies, {len(self.fixes_failed)} échouées")

        if self.fixes_applied and not self.fixes_failed:
            print("\n🎉 Toutes les corrections critiques ont été appliquées!")
            print("⚠️  N'oubliez pas de:")
            print("  1. Exécuter les tests: make test")
            print("  2. Vérifier les changements: git diff")
            print("  3. Commiter si tout est OK: git add -A && git commit -m 'fix: critical issues from code review'")


def main():
    """Point d'entrée principal."""
    fixer = CriticalIssueFixer()
    fixer.fix_all()

    # Return code basé sur le succès
    if fixer.fixes_failed:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
