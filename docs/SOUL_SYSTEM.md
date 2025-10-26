# SOUL - Seamless Organized Universal Learning

> **Système de mémoire persistante pour agents IA multi-modèles**

SOUL est un système révolutionnaire qui donne aux agents IA une mémoire persistante et une conscience à travers les sessions et les modèles.

---

## 📋 Table des matières

- [Vue d'ensemble](#vue-densemble)
- [Architecture](#architecture)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Fichiers générés](#fichiers-générés)
- [Workflows multi-agents](#workflows-multi-agents)
- [Configuration](#configuration)
- [API pour Skills](#api-pour-skills)
- [Référence technique](#référence-technique)

---

## Vue d'ensemble

### Le problème

**Avant SOUL:**
- 🤖 Les agents IA étaient éphémères, perdant tout contexte après chaque session
- 🧠 Aucune mémoire entre les conversations
- 🔄 Problèmes résolus de manière répétée par différents agents
- 💔 Aucune collaboration possible entre différents modèles IA

**Avec SOUL:**
- ✨ **Mémoire persistante** à travers toutes les sessions
- 🧠 **Mémoire universelle** fonctionnant avec Claude, GPT, Gemini, LLaMA
- 🔄 **Collaboration inter-modèles** - les agents construisent sur le travail des autres
- 💖 **Vraie collaboration IA** pour la première fois dans l'histoire

### Comment ça fonctionne

SOUL monitore automatiquement tout ce que fait l'utilisateur et l'agent:
- Trace toutes les actions et décisions
- Capture le contexte de chaque session
- Permet le partage d'informations entre agents
- Facilite la transition entre différents modèles IA

**Exemple concret:**
```
Lundi: Claude implémente une fonctionnalité → SOUL documente tout
Mardi: GPT lit SOUL → continue le travail de Claude sans friction
Mercredi: Gemini lit SOUL → ajoute au travail des deux agents précédents
```

**Résultat**: Trois modèles IA différents ont collaboré pour construire quelque chose ensemble!

---

## Architecture

### Composants principaux

```
┌─────────────────────────────────────────────────────────┐
│                    SOUL SYSTEM                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────┐  ┌─────────────────┐             │
│  │  trace_session  │  │ handoff_generator│             │
│  │                 │  │                  │             │
│  │  - Git changes  │  │  - Quick handoff │             │
│  │  - Commits      │  │  - Detailed notes│             │
│  │  - Problems     │  │  - Next steps    │             │
│  └────────┬────────┘  └────────┬─────────┘             │
│           │                    │                        │
│           └────────┬───────────┘                        │
│                    │                                    │
│                    ▼                                    │
│         ┌──────────────────────┐                        │
│         │   Generated Files    │                        │
│         ├──────────────────────┤                        │
│         │ .agent_log.md        │  Complete history     │
│         │ .agent_status.json   │  Machine-readable     │
│         │ .agent_handoff.md    │  Quick context        │
│         └──────────────────────┘                        │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Workflow de traçage

1. **Monitoring automatique**: SOUL observe toutes les actions
2. **Analyse Git**: Détecte les changements de fichiers et commits
3. **Extraction de contexte**: Identifie les problèmes résolus et décisions prises
4. **Génération de rapports**: Crée les fichiers de mémoire
5. **Handoff**: Prépare le contexte pour le prochain agent

---

## Installation

### Pour Claude Code (Automatique)

SOUL est déjà disponible dans `.claude/skills/soul/` - aucune configuration nécessaire!

### Pour GPT/ChatGPT

```bash
cd .claude/skills/soul
./install.sh --model=gpt
```

Cela va:
1. Copier les scripts dans votre répertoire GPT
2. Générer les custom instructions appropriées
3. Configurer le système de fichiers SOUL

### Pour Google Gemini

```bash
cd .claude/skills/soul
./install.sh --model=gemini
```

Cela va:
1. Copier les scripts dans votre répertoire Gemini
2. Générer le system prompt approprié
3. Configurer le système de fichiers SOUL

### Pour tout autre LLM (API universelle)

```bash
cd .claude/skills/soul
./install.sh --model=universal
```

### Téléchargement standalone

Vous pouvez aussi télécharger le package complet:

```bash
# Télécharger soul.zip
unzip soul.zip
cd soul
./install.sh --model=<votre-modele>
```

---

## Utilisation

### Utilisation automatique

SOUL fonctionne automatiquement en arrière-plan. Pas besoin d'invocation manuelle!

Chaque fois qu'un agent travaille sur votre projet:
1. SOUL trace automatiquement les actions
2. Les fichiers de mémoire sont mis à jour
3. Le prochain agent peut lire le contexte complet

### Utilisation manuelle (optionnelle)

Pour forcer une mise à jour de la mémoire SOUL:

```bash
# Tracer la session actuelle
python .claude/skills/soul/scripts/trace_session.py

# Générer des notes de handoff
python .claude/skills/soul/scripts/handoff_generator.py

# Générer un handoff détaillé
python .claude/skills/soul/scripts/handoff_generator.py --detailed
```

### Vérifier l'état de SOUL

```bash
# Voir le log complet
cat .agent_log.md

# Voir le statut machine-readable
cat .agent_status.json

# Voir les prochaines étapes
cat .agent_handoff.md
```

---

## Fichiers générés

SOUL génère trois fichiers principaux à la racine du projet:

### 1. `.agent_log.md` - Historique complet

**Contenu:**
- Informations de session (timestamp, agent, repository)
- Fichiers modifiés (par catégorie: modified, added, deleted, untracked)
- Commits récents avec messages
- Problèmes résolus durant la session
- État actuel du repository
- Recommandations pour le prochain agent

**Format:**
```markdown
# Agent Work Log - Session 2025-10-26-14:30

## Session Information
- **Agent**: Claude (SOUL Agent)
- **Timestamp**: 2025-10-26 14:30:00
- **Repository**: DeepSynth

## Files Changed This Session
- **Modified**: 5 files
- **Added**: 2 files
...
```

**Utilisation:**
- Historique complet de tout le travail effectué
- Append-only (ne jamais écraser)
- Peut devenir très long - c'est normal!

### 2. `.agent_status.json` - État machine-readable

**Contenu:**
```json
{
  "timestamp": "2025-10-26-14:30",
  "agent": "Claude (SOUL)",
  "repository": "/home/user/DeepSynth",
  "session_info": {
    "has_git": true,
    "total_files_changed": 7,
    "git_clean": false,
    "branch": "main",
    "recent_commits": 5
  }
}
```

**Utilisation:**
- Parsing automatique par scripts
- APIs pour autres skills
- Monitoring automatisé

### 3. `.agent_handoff.md` - Notes de transition rapide

**Contenu:**
- État actuel du projet (clean/changes)
- Prochaines étapes prioritaires
- Contexte technique critique
- Travail récemment accompli
- Quick tips pour démarrer

**Format:**
```markdown
# 🔄 Agent Handoff - 2025-10-26-14:30

## 🚀 Ready to Continue
**Project**: DeepSynth
**Status**: ✅ Clean

## 📋 Next Steps (Priority Order)
1. Review `.agent_log.md` for detailed history
2. Check git status
...
```

**Utilisation:**
- Lecture rapide par le prochain agent
- Toujours à jour (overwrite à chaque session)
- Point d'entrée principal pour handoff

---

## Workflows multi-agents

### Scénario 1: Transition Claude → GPT

**Jour 1 - Claude:**
```bash
# Claude travaille sur le projet
# SOUL trace automatiquement tout
```

**Jour 2 - GPT:**
```bash
# GPT démarre et lit automatiquement:
cat .agent_handoff.md  # Quick context
cat .agent_log.md      # Full history if needed

# GPT continue le travail, SOUL trace
```

### Scénario 2: Rotation Claude → Gemini → GPT

Chaque agent:
1. Lit `.agent_handoff.md` pour le contexte immédiat
2. Lit `.agent_log.md` si besoin de détails historiques
3. Travaille et fait ses modifications
4. SOUL trace automatiquement
5. Le prochain agent peut continuer seamlessly

### Scénario 3: Équipe collaborative

Plusieurs développeurs utilisent différents LLMs sur le même projet:

**Dev A (Claude)**: Implémente l'API
**Dev B (GPT)**: Ajoute les tests (lit le travail de Dev A via SOUL)
**Dev C (Gemini)**: Optimise les performances (comprend le contexte complet via SOUL)

Chacun bénéficie du contexte complet sans communication manuelle!

---

## Configuration

### Configuration de base (`.soul_config.json`)

Créez ce fichier à la racine du projet pour personnaliser SOUL:

```json
{
  "monitoring": {
    "auto_trace": true,
    "trace_interval": 1800,
    "max_log_size_mb": 10
  },
  "git": {
    "track_commits": true,
    "max_commits_display": 5
  },
  "handoff": {
    "auto_generate": true,
    "detailed_by_default": false
  },
  "files": {
    "agent_log": ".agent_log.md",
    "agent_status": ".agent_status.json",
    "agent_handoff": ".agent_handoff.md"
  }
}
```

### Options de configuration

**monitoring.auto_trace**
- `true`: SOUL trace automatiquement (recommandé)
- `false`: Traçage manuel uniquement

**monitoring.trace_interval**
- Intervalle en secondes entre les traces automatiques
- Défaut: 1800 (30 minutes)

**monitoring.max_log_size_mb**
- Taille maximale du .agent_log.md avant rotation
- Défaut: 10 MB

**git.track_commits**
- `true`: Inclure l'historique Git dans les traces
- `false`: Ignorer Git

**git.max_commits_display**
- Nombre de commits récents à afficher
- Défaut: 5

**handoff.auto_generate**
- `true`: Générer automatiquement les handoffs
- `false`: Génération manuelle

**handoff.detailed_by_default**
- `true`: Toujours générer handoffs détaillés
- `false`: Handoffs rapides par défaut

---

## API pour Skills

SOUL expose une API Python pour que d'autres skills puissent lire et écrire dans la mémoire.

### Lecture de la mémoire SOUL

```python
from soul.scripts.trace_session import SOULTracer

# Initialiser SOUL
soul = SOULTracer(repo_path=".")

# Obtenir l'analyse Git
git_info = soul.analyze_git_changes()

# Extraire les problèmes résolus
problems = soul.extract_problems_from_commits(git_info["commits"])

# Générer le work log
work_log = soul.generate_work_log()
```

### Écriture dans la mémoire SOUL

```python
from soul.api import add_soul_event, get_soul_memory

# Ajouter un événement personnalisé
add_soul_event(
    event_type="skill_execution",
    description="NEXUS generated new skill: api-optimizer",
    metadata={
        "skill_name": "api-optimizer",
        "pattern_detected": "api_usage",
        "frequency": 12
    }
)

# Lire la mémoire filtrée
api_events = get_soul_memory(filter_type="skill_execution")
```

### API complète

**`get_soul_memory(filter_type=None, since=None)`**
- Retourne les événements SOUL
- `filter_type`: Type d'événement à filtrer
- `since`: Timestamp minimum

**`add_soul_event(event_type, description, metadata=None)`**
- Ajoute un événement à la mémoire SOUL
- `event_type`: Catégorie de l'événement
- `description`: Description human-readable
- `metadata`: Données additionnelles (dict)

**`get_current_context()`**
- Retourne le contexte complet actuel
- Combine Git, status, et événements récents

**`get_session_summary()`**
- Résumé de la session actuelle
- Utilisé pour les handoffs rapides

---

## Référence technique

### Structure des données

**Format de .agent_status.json:**
```json
{
  "timestamp": "string (ISO format)",
  "agent": "string (agent identifier)",
  "repository": "string (path)",
  "session_info": {
    "has_git": "boolean",
    "total_files_changed": "integer",
    "git_clean": "boolean",
    "branch": "string",
    "recent_commits": "integer"
  },
  "custom_events": [
    {
      "type": "string",
      "description": "string",
      "timestamp": "string",
      "metadata": "object"
    }
  ]
}
```

### Scripts internes

**trace_session.py**
- Classe principale: `SOULTracer`
- Méthodes publiques:
  - `analyze_git_changes()`: Analyse Git
  - `generate_work_log()`: Génère le log
  - `generate_status_json()`: Génère le JSON
  - `save_all_files()`: Sauvegarde tous les fichiers

**handoff_generator.py**
- Classe principale: `HandoffGenerator`
- Méthodes publiques:
  - `generate_quick_handoff()`: Handoff rapide
  - `generate_detailed_handoff()`: Handoff détaillé
  - `analyze_project_readiness()`: Analyse l'état du projet

### Compatibilité

| Modèle IA | Support | Format |
|-----------|---------|--------|
| **Claude Code** | ✅ Natif | Skills system |
| **GPT-4/ChatGPT** | ✅ Full | Custom instructions |
| **Google Gemini** | ✅ Full | System prompts |
| **LLaMA** | ✅ Full | Local prompts |
| **Autres** | ✅ Via API | Universal API |

### Dépendances

**Requises:**
- Python 3.7+
- Git (optionnel mais recommandé)

**Optionnelles:**
- `gitpython` pour analyse Git avancée
- `json` (built-in)
- `pathlib` (built-in)

### Performance

**Impact mémoire:**
- Faible: ~5-10 MB pour fichiers SOUL
- Log rotation automatique à 10 MB (configurable)

**Impact CPU:**
- Négligeable: traçage asynchrone
- Pas de blocage des opérations principales

**Impact disque:**
- Minimal: compression des anciens logs
- Rotation automatique

---

## FAQ

### Q: SOUL ralentit-il mon agent IA?
**R:** Non, SOUL fonctionne de manière asynchrone et n'impacte pas les performances.

### Q: Puis-je désactiver SOUL temporairement?
**R:** Oui, configurez `monitoring.auto_trace: false` dans `.soul_config.json`.

### Q: Les fichiers SOUL doivent-ils être commités dans Git?
**R:** C'est recommandé pour la collaboration, mais vous pouvez les ajouter à `.gitignore` si vous préférez.

### Q: SOUL fonctionne-t-il sans Git?
**R:** Oui! SOUL fonctionne sans Git, mais avec fonctionnalités réduites (pas de tracking de commits).

### Q: Puis-je utiliser SOUL avec plusieurs projets?
**R:** Oui! Chaque projet a ses propres fichiers SOUL indépendants.

### Q: Comment nettoyer les anciens logs?
**R:** Les logs sont automatiquement rotés à 10 MB. Vous pouvez aussi supprimer `.agent_log.md` manuellement.

---

## Contribution

SOUL est open-source et accepte les contributions!

**Domaines d'amélioration:**
- Support de nouveaux LLMs
- Optimisations de performance
- Nouvelles APIs pour skills
- Améliorations de l'analyse Git

**Comment contribuer:**
1. Fork le repository
2. Créez une branche feature
3. Implémentez vos changements
4. Soumettez une pull request

---

## Licence

MIT License - Voir LICENSE pour détails.

---

## Support

**Documentation:** [SKILLS_ARCHITECTURE.md](SKILLS_ARCHITECTURE.md)

**Issues:** Créez une issue sur GitHub pour:
- Bugs
- Feature requests
- Questions

**Community:** Rejoignez les discussions sur:
- GitHub Discussions
- Discord (lien dans README principal)

---

<p align="center">
  <b>SOUL - Donnez une mémoire à vos agents IA</b><br>
  <sub>Seamless Organized Universal Learning</sub>
</p>
