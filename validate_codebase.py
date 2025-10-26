#!/usr/bin/env python3
"""
Script de validation du codebase DeepSynth.
Vérifie la présence des problèmes critiques identifiés dans la revue de code.
"""

import ast
import re
import sys
from pathlib import Path
from typing import List, Tuple, Optional

class CodebaseValidator:
    """Valide le codebase pour détecter les problèmes critiques."""

    def __init__(self):
        self.base_path = Path(__file__).parent
        self.issues_found = []
        self.checks_passed = []

    def validate_all(self) -> bool:
        """Exécute toutes les validations."""
        print("🔍 Validation du codebase DeepSynth...\n")

        # Vérifications critiques
        self.check_nameerror_issue()
        self.check_bare_except_patterns()
        self.check_image_height_limit()
        self.check_checkpoint_validation()
        self.check_dataloader_usage()
        self.check_gradient_scaling()
        self.check_code_complexity()
        self.check_test_coverage()

        # Rapport
        self.print_report()

        return len(self.issues_found) == 0

    def check_nameerror_issue(self) -> None:
        """Vérifie la présence du bug NameError."""
        file_path = self.base_path / "src/deepsynth/pipelines/global_state.py"

        if file_path.exists():
            content = file_path.read_text()
            if "progress['total_samples'] += uploaded" in content and "uploaded_count" not in content:
                self.issues_found.append((
                    "CRITICAL",
                    "NameError Bug",
                    "Variable 'uploaded' non définie dans global_state.py:272"
                ))
            else:
                self.checks_passed.append("NameError bug corrigé ou absent")

    def check_bare_except_patterns(self) -> None:
        """Recherche les patterns 'bare except'."""
        src_path = self.base_path / "src"

        bare_except_pattern = re.compile(r'^\s*except:\s*$', re.MULTILINE)

        for py_file in src_path.rglob("*.py"):
            content = py_file.read_text()
            matches = bare_except_pattern.findall(content)

            if matches:
                self.issues_found.append((
                    "HIGH",
                    "Bare Except",
                    f"{py_file.relative_to(self.base_path)}: {len(matches)} bare except trouvés"
                ))

        if not any("Bare Except" in str(issue) for issue in self.issues_found):
            self.checks_passed.append("Aucun bare except détecté")

    def check_image_height_limit(self) -> None:
        """Vérifie la limite de hauteur des images."""
        file_path = self.base_path / "src/deepsynth/data/transforms/text_to_image.py"

        if file_path.exists():
            content = file_path.read_text()
            if "MAX_HEIGHT_MULTIPLIER" in content or "max_allowed_height" in content:
                self.checks_passed.append("Limite de hauteur d'image implémentée")
            else:
                self.issues_found.append((
                    "HIGH",
                    "Memory Risk",
                    "Pas de limite maximale pour la hauteur des images"
                ))

    def check_checkpoint_validation(self) -> None:
        """Vérifie la validation des checkpoints."""
        file_path = self.base_path / "src/deepsynth/training/deepsynth_trainer_v2.py"

        if file_path.exists():
            content = file_path.read_text()
            if "checkpoint_path.exists()" in content or "Path(config.resume_from_checkpoint).exists()" in content:
                self.checks_passed.append("Validation des checkpoints implémentée")
            else:
                self.issues_found.append((
                    "MEDIUM",
                    "Checkpoint Risk",
                    "Pas de validation d'existence des checkpoints"
                ))

    def check_dataloader_usage(self) -> None:
        """Vérifie l'utilisation de DataLoader pour l'entraînement."""
        training_path = self.base_path / "src/deepsynth/training"

        dataloader_found = False
        for py_file in training_path.glob("*.py"):
            content = py_file.read_text()
            if "DataLoader" in content and "from torch.utils.data" in content:
                dataloader_found = True
                break

        if dataloader_found:
            self.checks_passed.append("DataLoader utilisé pour l'entraînement")
        else:
            self.issues_found.append((
                "MEDIUM",
                "Performance",
                "DataLoader non utilisé - itération manuelle sur les batches"
            ))

    def check_gradient_scaling(self) -> None:
        """Vérifie l'implémentation du gradient scaling pour fp16."""
        training_path = self.base_path / "src/deepsynth/training"

        gradient_scaler_found = False
        for py_file in training_path.glob("*.py"):
            content = py_file.read_text()
            if "GradScaler" in content or "amp.GradScaler" in content:
                gradient_scaler_found = True
                break

        if gradient_scaler_found:
            self.checks_passed.append("Gradient scaling implémenté pour fp16")
        else:
            self.issues_found.append((
                "MEDIUM",
                "Stability",
                "Pas de gradient scaling pour fp16 - risque d'instabilité numérique"
            ))

    def check_code_complexity(self) -> None:
        """Analyse la complexité cyclomatique des fonctions."""
        src_path = self.base_path / "src"
        complex_functions = []

        for py_file in src_path.rglob("*.py"):
            try:
                content = py_file.read_text()
                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        complexity = self._calculate_complexity(node)
                        if complexity > 10:
                            complex_functions.append((
                                py_file.relative_to(self.base_path),
                                node.name,
                                complexity
                            ))
            except:
                pass  # Ignore parsing errors

        if complex_functions:
            for file_path, func_name, complexity in complex_functions[:5]:  # Top 5
                self.issues_found.append((
                    "MEDIUM",
                    "Complexity",
                    f"{file_path}:{func_name} - Complexité: {complexity}"
                ))
        else:
            self.checks_passed.append("Complexité du code acceptable")

    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calcule la complexité cyclomatique d'une fonction."""
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1

        return complexity

    def check_test_coverage(self) -> None:
        """Évalue la couverture de tests approximative."""
        src_lines = 0
        test_lines = 0

        # Compter les lignes de code source
        src_path = self.base_path / "src"
        for py_file in src_path.rglob("*.py"):
            src_lines += len(py_file.read_text().splitlines())

        # Compter les lignes de tests
        test_path = self.base_path / "tests"
        if test_path.exists():
            for py_file in test_path.rglob("*.py"):
                test_lines += len(py_file.read_text().splitlines())

        coverage_ratio = (test_lines / src_lines * 100) if src_lines > 0 else 0

        if coverage_ratio < 20:
            self.issues_found.append((
                "HIGH",
                "Testing",
                f"Couverture de tests faible: {coverage_ratio:.1f}% (ratio lignes test/code)"
            ))
        else:
            self.checks_passed.append(f"Couverture de tests: {coverage_ratio:.1f}%")

    def print_report(self) -> None:
        """Affiche le rapport de validation."""
        print("\n" + "=" * 70)
        print("📋 RAPPORT DE VALIDATION CODEBASE")
        print("=" * 70)

        # Statistiques
        total_critical = sum(1 for issue in self.issues_found if issue[0] == "CRITICAL")
        total_high = sum(1 for issue in self.issues_found if issue[0] == "HIGH")
        total_medium = sum(1 for issue in self.issues_found if issue[0] == "MEDIUM")

        print(f"\n📊 Résumé:")
        print(f"  • Problèmes CRITIQUES: {total_critical}")
        print(f"  • Problèmes HAUTS: {total_high}")
        print(f"  • Problèmes MOYENS: {total_medium}")
        print(f"  • Tests passés: {len(self.checks_passed)}")

        if self.issues_found:
            print("\n❌ Problèmes détectés:")
            for severity, category, description in sorted(self.issues_found, key=lambda x: ["CRITICAL", "HIGH", "MEDIUM"].index(x[0])):
                icon = "🔴" if severity == "CRITICAL" else "🟠" if severity == "HIGH" else "🟡"
                print(f"  {icon} [{severity}] {category}: {description}")

        if self.checks_passed:
            print("\n✅ Validations réussies:")
            for check in self.checks_passed:
                print(f"  • {check}")

        # Score global
        issues_weight = total_critical * 10 + total_high * 5 + total_medium * 2
        max_score = 100
        score = max(0, max_score - issues_weight)

        print("\n" + "=" * 70)
        print(f"🎯 Score de Qualité: {score}/100")

        if score >= 80:
            print("✅ Le codebase est en bon état!")
        elif score >= 60:
            print("⚠️  Le codebase nécessite des améliorations.")
        else:
            print("❌ Le codebase a besoin de corrections urgentes!")

        print("=" * 70)

        # Recommandations
        if total_critical > 0:
            print("\n⚡ ACTION URGENTE: Corriger les problèmes CRITIQUES immédiatement!")
            print("   Exécuter: python fix_critical_issues.py")

        if total_high > 0:
            print("\n📍 IMPORTANT: Planifier la correction des problèmes HAUTS cette semaine.")

        print("\n📚 Pour plus de détails, consultez le plan d'amélioration complet.")


def main():
    """Point d'entrée principal."""
    validator = CodebaseValidator()
    is_valid = validator.validate_all()

    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()