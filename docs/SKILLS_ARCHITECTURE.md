# Architecture du Système de Skills Claude

> **Système modulaire de skills IA avec mémoire persistante et génération automatique**

Ce document décrit l'architecture complète du système de skills pour Claude Code, incluant SOUL, NEXUS et PRD-TASKMASTER.

---

## 📋 Table des matières

- [Vue d'ensemble](#vue-densemble)
- [Architecture globale](#architecture-globale)
- [Les 3 skills principaux](#les-3-skills-principaux)
  - [SOUL - Mémoire universelle](#soul---mémoire-universelle)
  - [NEXUS - Générateur de skills](#nexus---générateur-de-skills)
  - [PRD-TASKMASTER - Analyseur de tâches](#prd-taskmaster---analyseur-de-tâches)
- [Workflow complet](#workflow-complet)
- [Compatibilité multi-LLM](#compatibilité-multi-llm)
- [Guide de développement](#guide-de-développement)
- [Distribution et installation](#distribution-et-installation)

---

## Vue d'ensemble

### Problème résolu

Les agents IA traditionnels souffrent de plusieurs limitations:
- **Pas de mémoire persistante** entre sessions
- **Pas de collaboration** entre différents modèles IA
- **Skills trop spécifiques** qui ne s'adaptent pas aux besoins évolutifs
- **Réinvention de la roue** pour chaque nouveau projet

### Notre solution

Un système de **3 skills génériques et interconnectés** qui:
1. **SOUL**: Trace et mémorise tout ce qui se passe
2. **NEXUS**: Observe les patterns et génère automatiquement de nouveaux skills
3. **PRD-TASKMASTER**: Analyse les besoins du projet et recommande les skills appropriés

### Principes de design

**Généricité**: Pas de skills spécifiques (API, database, etc.) - tout est généré à la demande

**Universalité**: Compatible avec Claude, GPT, Gemini et tout autre LLM

**Automatisation**: Les skills se génèrent automatiquement selon les besoins détectés

**Mémoire**: Contexte persistant à travers sessions et modèles

---

## Architecture globale

```
┌─────────────────────────────────────────────────────────────────────┐
│                         UTILISATEUR                                 │
│                  (demande nouvelle fonctionnalité)                  │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │        SOUL            │
                    │  (Mémoire universelle) │
                    │                        │
                    │  • Trace actions       │
                    │  • Enregistre contexte │
                    │  • Partage mémoire     │
                    │  • Handoff agents      │
                    └──────────┬─────────────┘
                               │
                ┌──────────────┴─────────────┐
                │                            │
                ▼                            ▼
    ┌───────────────────┐        ┌──────────────────────┐
    │  PRD-TASKMASTER   │        │       NEXUS          │
    │ (Analyse besoins) │        │ (Observe patterns)   │
    │                   │        │                      │
    │  • Parse PRD      │        │  • Lit SOUL          │
    │  • Extrait tasks  │        │  • Détecte patterns  │
    │  • Classifie      │        │  • Pattern ≥ 5×      │
    │  • Recommande     │        │  • Suggère skills    │
    └─────────┬─────────┘        └──────────┬───────────┘
              │                             │
              └──────────────┬──────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │     NEXUS       │
                    │ (Génère skills) │
                    │                 │
                    │  • Templates    │
                    │  • SKILL.md     │
                    │  • Scripts      │
                    │  • Multi-LLM    │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  NOUVEAU SKILL  │
                    │  (automatique)  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │      SOUL       │
                    │ (utilise skill) │
                    │ (trace usage)   │
                    └─────────────────┘
```

### Flux de données

1. **User → SOUL**: Toutes les actions utilisateur sont tracées
2. **SOUL → PRD-TASKMASTER**: Analyse des PRD et tâches du projet
3. **SOUL → NEXUS**: Partage de l'historique pour détection de patterns
4. **PRD-TASKMASTER → NEXUS**: Recommandations de skills basées sur l'analyse
5. **NEXUS → Nouveau Skill**: Génération automatique du skill
6. **Nouveau Skill → SOUL**: Le skill généré utilise SOUL pour sa mémoire

---

## Les 3 skills principaux

### SOUL - Mémoire universelle

**Rôle**: Système de mémoire persistante qui trace tout ce que fait l'utilisateur et permet le partage d'information entre agents et skills.

**Localisation**: `.claude/skills/soul/`

#### Structure

```
soul/
├── SKILL.md                    # Description du skill
├── README.md                   # Guide d'installation
├── scripts/
│   ├── trace_session.py        # Traçage des sessions
│   ├── handoff_generator.py    # Génération de handoffs
│   ├── soul_api.py            # API pour autres skills
│   └── install.sh              # Installation multi-LLM
├── claude/
│   └── skill.md                # Format Claude Code
├── gpt/
│   └── custom_instructions.md  # Format GPT
├── gemini/
│   └── system_prompt.md        # Format Gemini
└── soul.zip                    # Package de distribution
```

#### Fonctionnalités principales

**Monitoring complet:**
- Trace toutes les actions utilisateur
- Capture les décisions importantes
- Enregistre l'historique des skills utilisés
- Maintient le contexte inter-sessions

**APIs pour skills:**
```python
# Lire la mémoire SOUL
get_soul_memory(filter_type=None, since=None)

# Écrire dans SOUL
add_soul_event(event_type, description, metadata=None)

# Contexte actuel
get_current_context()

# Résumé de session
get_session_summary()
```

**Fichiers générés:**
- `.agent_log.md` - Historique complet et détaillé
- `.agent_status.json` - État machine-readable
- `.agent_handoff.md` - Notes de transition rapides

#### Cas d'usage

**Scénario 1 - Collaboration multi-agents:**
```
Lundi (Claude):    Implémente API REST
                   → SOUL trace tout

Mardi (GPT):       Lit SOUL, comprend le contexte
                   → Ajoute les tests
                   → SOUL trace

Mercredi (Gemini): Lit SOUL, voit travail complet
                   → Optimise les performances
                   → SOUL trace
```

**Scénario 2 - Partage de contexte entre skills:**
```
Skill A (API-Generator):  Génère des endpoints
                         → add_soul_event("api_generated", ...)

Skill B (Test-Generator): Lit get_soul_memory(filter_type="api_generated")
                         → Génère tests pour ces endpoints
```

📖 **[Documentation complète SOUL](SOUL_SYSTEM.md)**

---

### NEXUS - Générateur de skills

**Rôle**: Observer l'historique SOUL et les patterns d'utilisation, puis générer automatiquement de nouveaux skills quand nécessaire.

**Localisation**: `.claude/skills/nexus/`

#### Structure

```
nexus/
├── SKILL.md                           # Description du skill
├── README.md                          # Guide d'installation
├── scripts/
│   ├── skill_generator.py             # Générateur principal
│   ├── pattern_detector.py            # Détection de patterns
│   ├── soul_integration.py            # Intégration SOUL
│   ├── skill_templates.py             # Templates génériques
│   └── install.sh                     # Installation multi-LLM
├── templates/
│   ├── generic_skill_template/        # Template de base
│   │   ├── SKILL.md
│   │   ├── scripts/
│   │   └── README.md
│   └── skill_config_template.json     # Configuration par défaut
├── claude/
│   └── skill.md
├── gpt/
│   └── custom_instructions.md
├── gemini/
│   └── system_prompt.md
└── nexus.zip                          # Package de distribution
```

#### Fonctionnalités principales

**Détection de patterns:**
- Lit l'historique SOUL périodiquement
- Compte les occurrences de patterns similaires
- Seuil de déclenchement: 5+ occurrences
- Évite la duplication avec skills existants

**Génération automatique:**
- Crée la structure complète du skill
- Génère SKILL.md avec description
- Crée les scripts Python appropriés
- Configure pour multi-LLM (Claude/GPT/Gemini)

**Intégration SOUL:**
```python
from soul_integration import get_soul_patterns

# Obtenir les patterns détectés par SOUL
patterns = get_soul_patterns(threshold=5, days=7)

for pattern in patterns:
    if should_generate_skill(pattern):
        generate_skill(
            name=pattern.suggested_name,
            type=pattern.pattern_type,
            context=pattern.context
        )
```

**Configuration:**
```json
{
  "pattern_detection": {
    "threshold": 5,
    "window_days": 7,
    "auto_generate": true
  },
  "skill_generation": {
    "naming_style": "creative",
    "complexity": "comprehensive",
    "include_examples": true
  }
}
```

#### Workflow de génération

1. **Détection**: NEXUS lit SOUL toutes les 6 heures (configurable)
2. **Analyse**: Identifie les patterns récurrents (API calls, data processing, etc.)
3. **Validation**: Vérifie qu'un skill similaire n'existe pas déjà
4. **Génération**: Crée le skill complet avec templates
5. **Notification**: Informe l'utilisateur et SOUL du nouveau skill
6. **Activation**: Le skill est immédiatement disponible

#### Exemples de patterns détectés

**Pattern API Usage (détecté 8 fois):**
```
→ NEXUS génère "api-optimizer" skill
   • Rate limiting automatique
   • Retry logic
   • Error handling
   • Response caching
```

**Pattern Data Processing (détecté 12 fois):**
```
→ NEXUS génère "data-transformer" skill
   • CSV/JSON parsing
   • Validation
   • Transformation pipelines
   • Format conversion
```

---

### PRD-TASKMASTER - Analyseur de tâches

**Rôle**: Analyser les PRD (Product Requirement Documents) et les listes de tâches, puis générer des recommandations de skills pour NEXUS.

**Localisation**: `.claude/skills/prd-taskmaster/`

#### Structure

```
prd-taskmaster/
├── SKILL.md                          # Description du skill
├── README.md                         # Guide d'installation
├── scripts/
│   ├── prd_analyzer.py               # Analyseur principal
│   ├── task_classifier.py            # Classification des tâches
│   ├── nexus_integration.py          # Intégration NEXUS
│   └── install.sh                    # Installation multi-LLM
├── config/
│   └── domain_patterns.json          # Patterns de domaines
├── claude/
│   └── skill.md
├── gpt/
│   └── custom_instructions.md
├── gemini/
│   └── system_prompt.md
└── prd-taskmaster.zip                # Package de distribution
```

#### Fonctionnalités principales

**Recherche de PRD:**
- Scanne le projet pour trouver les fichiers PRD
- Patterns: `*PRD*.md`, `*TODO*.md`, `*TASK*.md`, `*REQUIREMENTS*.md`
- Détection intelligente dans `docs/`, racine, etc.

**Extraction de tâches:**
- Checkbox tasks: `- [ ] Task` ou `- [x] Task`
- Listes numérotées: `1. Task`
- Bullet points: `- Task` ou `* Task`
- Headers de tâches: `### Implement feature`

**Classification par domaine:**
```python
DOMAIN_PATTERNS = {
    "api": ["api", "endpoint", "rest", "graphql", "request"],
    "testing": ["test", "testing", "unit test", "coverage"],
    "deployment": ["deploy", "docker", "kubernetes", "ci/cd"],
    "documentation": ["readme", "docs", "documentation"],
    "database": ["database", "sql", "query", "migration"],
    "performance": ["performance", "optimize", "cache"],
    "security": ["security", "auth", "encrypt", "permission"],
    "data_processing": ["data", "etl", "transform", "parse"]
}
```

**Recommandations de skills:**
- Analyse les tâches classifiées
- Calcule les priorités (critical/high/medium/low)
- Génère des recommandations pour NEXUS
- Fournit les capabilities suggérées pour chaque skill

#### Workflow d'analyse

1. **Scan**: Recherche des fichiers PRD dans le projet
2. **Parse**: Extrait toutes les tâches des fichiers
3. **Classify**: Classifie chaque tâche par domaine
4. **Cluster**: Regroupe les tâches similaires
5. **Recommend**: Génère des recommandations de skills
6. **Notify NEXUS**: Envoie les recommandations à NEXUS pour génération

#### Intégration avec NEXUS

```python
from nexus_integration import send_skill_recommendations

# Analyse du PRD
analyzer = PRDAnalyzer()
analyzer.find_prd_files()
analyzer.parse_all_files()
patterns = analyzer.analyze_task_patterns()

# Génération de recommandations
recommendations = analyzer.generate_skill_recommendations()

# Envoi à NEXUS
send_skill_recommendations(recommendations)

# NEXUS génère automatiquement les skills recommandés
```

#### Rapport d'analyse

Génère un rapport JSON complet:
```json
{
  "total_tasks": 47,
  "summary": {
    "total_patterns_detected": 6,
    "skills_recommended": 4,
    "high_priority_skills": 2
  },
  "skill_recommendations": [
    {
      "domain": "api",
      "skill_name": "api-master",
      "priority": "critical",
      "task_count": 12,
      "reason": "Detected 12 tasks related to api",
      "recommended_capabilities": [
        "Rate limiting and retry logic",
        "Error handling patterns",
        "Response caching"
      ]
    }
  ]
}
```

---

## Workflow complet

### Scénario 1: Nouveau projet

```
Jour 1 - Initialisation:
  User: Crée un PRD avec 50 tâches
  ↓
  PRD-TASKMASTER: Scanne et analyse le PRD
    → Détecte 12 tâches API
    → Détecte 8 tâches testing
    → Détecte 5 tâches deployment
  ↓
  PRD-TASKMASTER → NEXUS: Envoie recommandations
  ↓
  NEXUS: Génère 3 skills automatiquement
    → api-handler (12 tâches API)
    → test-guardian (8 tâches testing)
    → deploy-sage (5 tâches deployment)
  ↓
  SOUL: Trace tout le processus
    → Enregistre les skills générés
    → Prêt pour utilisation
```

### Scénario 2: Projet en cours

```
Semaine 1-2:
  User: Travaille sur le projet avec Claude
  ↓
  SOUL: Trace toutes les actions
    → 15× appels API externes détectés
    → 8× transformations de données détectées
  ↓
  NEXUS (analyse automatique toutes les 6h):
    → Pattern API ≥ 5× → Génère "api-optimizer"
    → Pattern data ≥ 5× → Génère "data-transformer"
  ↓
  Les nouveaux skills sont disponibles immédiatement
  ↓
  SOUL: Utilise les nouveaux skills automatiquement
```

### Scénario 3: Collaboration multi-agents

```
Lundi (Claude + SOUL):
  User: Demande implémentation d'une API
  ↓
  Claude: Implémente l'API
  ↓
  SOUL: Trace l'implémentation
    → Fichiers créés
    → Endpoints implémentés
    → Décisions techniques

Mardi (GPT + SOUL):
  GPT: Lit .agent_handoff.md
  ↓
  Comprend le contexte complet de l'API
  ↓
  GPT: Ajoute les tests
  ↓
  SOUL: Trace les tests ajoutés

Mercredi (Gemini + SOUL + NEXUS):
  Gemini: Lit .agent_handoff.md et .agent_log.md
  ↓
  NEXUS: Détecte le pattern API (répété)
  ↓
  NEXUS: Génère "api-test-generator" skill
  ↓
  Gemini: Utilise le nouveau skill pour optimiser
```

---

## Compatibilité multi-LLM

### Structure universelle

Chaque skill est organisé pour supporter plusieurs LLMs:

```
skill-name/
├── SKILL.md                    # Description universelle
├── README.md                   # Guide d'installation
├── scripts/                    # Logique Python (universel)
│   ├── core_logic.py
│   └── install.sh
├── claude/                     # Spécifique Claude Code
│   └── skill.md
├── gpt/                        # Spécifique GPT
│   └── custom_instructions.md
├── gemini/                     # Spécifique Gemini
│   └── system_prompt.md
└── skill-name.zip              # Distribution
```

### Installation par modèle

**Claude Code:**
```bash
# Automatique - déjà dans .claude/skills/
# Aucune action nécessaire
```

**GPT/ChatGPT:**
```bash
cd .claude/skills/skill-name
./install.sh --model=gpt

# Génère custom instructions
# Affiche les instructions à copier dans ChatGPT
```

**Google Gemini:**
```bash
cd .claude/skills/skill-name
./install.sh --model=gemini

# Génère system prompt
# Affiche le prompt à configurer dans Gemini
```

**API universelle:**
```bash
cd .claude/skills/skill-name
./install.sh --model=universal

# Configure l'API REST
# Utilisable par n'importe quel LLM via HTTP
```

### Formats spécifiques

**Claude Code (skill.md):**
```markdown
---
name: skill-name
description: Description automatique du skill
---

# Skill content optimized for Claude Code skills system
```

**GPT (custom_instructions.md):**
```markdown
# Custom Instructions for skill-name

## What would you like ChatGPT to know about you?
[Context about the skill]

## How would you like ChatGPT to respond?
[Behavior instructions]
```

**Gemini (system_prompt.md):**
```markdown
# System Prompt for skill-name

You are an AI assistant with the skill-name capability.

[Detailed instructions for Gemini]
```

### Tableau de compatibilité

| Fonctionnalité | Claude | GPT-4 | Gemini | Autres |
|----------------|--------|-------|--------|--------|
| **SOUL** | ✅ Natif | ✅ Full | ✅ Full | ✅ API |
| **NEXUS** | ✅ Natif | ✅ Full | ✅ Full | ✅ API |
| **PRD-TASKMASTER** | ✅ Natif | ✅ Full | ✅ Full | ✅ API |
| **Skills générés** | ✅ Auto | ⚙️ Config | ⚙️ Config | 🔌 API |
| **Mémoire persistante** | ✅ Fichiers | ✅ Fichiers | ✅ Fichiers | ✅ Fichiers |
| **Auto-génération** | ✅ Oui | ⚠️ Manuel | ⚠️ Manuel | 🔌 API |

**Légende:**
- ✅ Support complet natif
- ⚙️ Nécessite configuration
- ⚠️ Support partiel (génération manuelle)
- 🔌 Via API REST

---

## Guide de développement

### Créer un nouveau skill (manuel)

Si NEXUS ne génère pas automatiquement le skill dont vous avez besoin, vous pouvez le créer manuellement:

#### 1. Structure de base

```bash
mkdir -p .claude/skills/mon-skill/{scripts,claude,gpt,gemini}
cd .claude/skills/mon-skill
```

#### 2. Créer SKILL.md

```markdown
---
name: mon-skill
description: Description courte de ce que fait le skill
---

# Mon Skill - Titre descriptif

## Ce que fait le skill
[Description détaillée]

## Quand Claude active ce skill
[Triggers d'activation]

## Capacités principales
[Liste des fonctionnalités]
```

#### 3. Créer le script principal

```python
# scripts/main.py
#!/usr/bin/env python3
"""
Mon Skill - Description
"""

import sys
from pathlib import Path

# Ajouter le path SOUL pour utiliser son API
sys.path.append(str(Path(__file__).parent.parent.parent / "soul" / "scripts"))

from soul_api import get_soul_memory, add_soul_event

class MonSkill:
    def __init__(self):
        self.name = "mon-skill"

    def execute(self, context):
        # Lire la mémoire SOUL si nécessaire
        history = get_soul_memory(filter_type="relevant_type")

        # Logique du skill
        result = self.do_something(context)

        # Enregistrer dans SOUL
        add_soul_event(
            event_type="mon_skill_execution",
            description=f"Mon skill a été exécuté avec succès",
            metadata={"result": result}
        )

        return result

    def do_something(self, context):
        # Votre logique ici
        return "success"

if __name__ == "__main__":
    skill = MonSkill()
    skill.execute({"some": "context"})
```

#### 4. Créer install.sh

```bash
#!/bin/bash
# Installation script for mon-skill

MODEL=${1:---model=claude}

case $MODEL in
  --model=claude)
    echo "Installing for Claude Code..."
    # Déjà dans .claude/skills/, rien à faire
    ;;
  --model=gpt)
    echo "Installing for GPT..."
    echo "Copy the content of gpt/custom_instructions.md"
    cat gpt/custom_instructions.md
    ;;
  --model=gemini)
    echo "Installing for Gemini..."
    echo "Use the following system prompt:"
    cat gemini/system_prompt.md
    ;;
  *)
    echo "Usage: ./install.sh --model=[claude|gpt|gemini|universal]"
    exit 1
    ;;
esac

echo "✓ mon-skill installation complete!"
```

#### 5. Adapter pour chaque LLM

**claude/skill.md**: Format Claude Code skills

**gpt/custom_instructions.md**: Instructions pour ChatGPT

**gemini/system_prompt.md**: System prompt pour Gemini

#### 6. Créer le zip de distribution

```bash
cd .claude/skills
zip -r mon-skill.zip mon-skill/ -x "*.pyc" "__pycache__/*" "*.zip"
```

### Utiliser l'API SOUL dans un skill

Tous les skills doivent utiliser SOUL pour la mémoire:

```python
from soul_api import (
    get_soul_memory,      # Lire la mémoire
    add_soul_event,       # Écrire un événement
    get_current_context,  # Contexte actuel
    get_session_summary   # Résumé de session
)

# Exemple: Skill qui utilise l'historique
def my_skill_logic():
    # Lire les événements API des 7 derniers jours
    api_events = get_soul_memory(
        filter_type="api_call",
        since=datetime.now() - timedelta(days=7)
    )

    # Analyser les patterns
    endpoints = analyze_api_patterns(api_events)

    # Enregistrer le résultat
    add_soul_event(
        event_type="api_analysis",
        description=f"Analyzed {len(endpoints)} API endpoints",
        metadata={"endpoints": endpoints}
    )
```

### Guidelines de développement

**1. Généricité**
- Éviter les skills trop spécifiques
- Préférer des skills configurables et adaptables
- Utiliser des templates et patterns

**2. Intégration SOUL**
- Toujours utiliser l'API SOUL pour la mémoire
- Enregistrer les événements importants
- Lire le contexte avant d'agir

**3. Multi-LLM**
- Séparer la logique (Python) de l'interface (markdown)
- Fournir des versions pour Claude/GPT/Gemini
- Tester sur plusieurs modèles

**4. Documentation**
- README.md avec installation claire
- SKILL.md avec description détaillée
- Exemples d'utilisation concrets

**5. Distribution**
- Créer un zip standalone
- Include install.sh avec --model flag
- Pas de dépendances externes lourdes

---

## Distribution et installation

### Packages de distribution

Chaque skill principal est distribué sous forme de zip:

**soul.zip** (~ 50 KB)
- Système de mémoire complet
- Scripts Python
- Adaptateurs multi-LLM
- Documentation

**nexus.zip** (~ 80 KB)
- Générateur de skills
- Templates
- Détection de patterns
- Documentation

**prd-taskmaster.zip** (~ 40 KB)
- Analyseur de PRD
- Classification de tâches
- Intégration NEXUS
- Documentation

### Installation depuis zip

**Téléchargement:**
```bash
# Depuis le repository
wget https://github.com/bacoco/deepseek-synthesia/raw/main/.claude/skills/soul.zip

# Ou depuis releases
curl -L -o soul.zip https://github.com/bacoco/deepseek-synthesia/releases/latest/download/soul.zip
```

**Extraction et installation:**
```bash
# Extraire
unzip soul.zip
cd soul/

# Installer pour votre LLM
./install.sh --model=claude    # Pour Claude Code
./install.sh --model=gpt       # Pour ChatGPT
./install.sh --model=gemini    # Pour Gemini
./install.sh --model=universal # API universelle
```

### Installation depuis source

**Clone du repository:**
```bash
git clone https://github.com/bacoco/deepseek-synthesia.git
cd deepseek-synthesia
```

**Utilisation directe:**
```bash
# Les skills sont déjà dans .claude/skills/
# Pour Claude Code, c'est automatique!

# Pour autres LLMs:
cd .claude/skills/soul
./install.sh --model=gpt
```

### Génération des zips (pour mainteneurs)

```bash
# Script de build
./scripts/build_distributions.sh

# Ou manuellement pour un skill:
cd .claude/skills/soul
zip -r ../soul.zip . -x "*.pyc" "__pycache__/*" "*.zip"
```

---

## Résumé

### Points clés

✅ **3 skills génériques** au lieu de dizaines de skills spécifiques

✅ **Génération automatique** de nouveaux skills selon les besoins

✅ **Mémoire persistante** à travers sessions et modèles

✅ **Compatibilité multi-LLM** (Claude, GPT, Gemini)

✅ **Architecture ouverte** et extensible

### Bénéfices

**Pour les utilisateurs:**
- Pas de configuration manuelle de skills
- Adaptation automatique aux besoins du projet
- Mémoire persistante entre sessions
- Compatible avec leur LLM préféré

**Pour les développeurs:**
- Architecture claire et modulaire
- APIs bien définies
- Templates réutilisables
- Documentation complète

**Pour les projets:**
- Skills adaptés au contexte spécifique
- Pas de bloat de skills inutilisés
- Évolution automatique avec le projet
- Collaboration multi-agents fluide

---

## Ressources

**Documentation:**
- [SOUL System](SOUL_SYSTEM.md) - Guide complet SOUL
- [README.md](../README.md) - Overview du projet

**Code source:**
- [.claude/skills/soul/](.claude/skills/soul/) - Code SOUL
- [.claude/skills/nexus/](.claude/skills/nexus/) - Code NEXUS
- [.claude/skills/prd-taskmaster/](.claude/skills/prd-taskmaster/) - Code PRD-TASKMASTER

**Distribution:**
- soul.zip - Package SOUL
- nexus.zip - Package NEXUS
- prd-taskmaster.zip - Package PRD-TASKMASTER

---

<p align="center">
  <b>Architecture de Skills Claude - Système générique et auto-adaptatif</b><br>
  <sub>SOUL • NEXUS • PRD-TASKMASTER</sub>
</p>
