#!/usr/bin/env python3
"""
SOUL API - Interface pour que les autres skills puissent lire/écrire dans la mémoire SOUL
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import fcntl

class SOULMemory:
    """Interface principale pour accéder à la mémoire SOUL"""

    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path).resolve()
        self.status_file = self.repo_path / ".agent_status.json"
        self.log_file = self.repo_path / ".agent_log.md"
        self.handoff_file = self.repo_path / ".agent_handoff.md"

        # Ensure files exist
        self._ensure_files_exist()

    def _ensure_files_exist(self):
        """Créer les fichiers SOUL s'ils n'existent pas"""
        if not self.status_file.exists():
            self._init_status_file()
        if not self.log_file.exists():
            self.log_file.touch()
        if not self.handoff_file.exists():
            self.handoff_file.touch()

    def _init_status_file(self):
        """Initialiser le fichier de statut"""
        initial_status = {
            "timestamp": datetime.now().strftime("%Y-%m-%d-%H:%M"),
            "agent": "SOUL System",
            "repository": str(self.repo_path),
            "session_info": {
                "has_git": False,
                "total_files_changed": 0,
                "git_clean": True
            },
            "custom_events": []
        }
        with open(self.status_file, 'w') as f:
            json.dump(initial_status, f, indent=2)

    def _load_status(self) -> Dict[str, Any]:
        """Charger le fichier de statut avec lock"""
        if not self.status_file.exists():
            self._init_status_file()

        with open(self.status_file, 'r') as f:
            # Lock pour lecture
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                data = json.load(f)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            return data

    def _save_status(self, status: Dict[str, Any]):
        """Sauvegarder le fichier de statut avec lock"""
        with open(self.status_file, 'w') as f:
            # Lock pour écriture
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(status, f, indent=2)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def add_event(self, event_type: str, description: str, metadata: Optional[Dict] = None):
        """
        Ajouter un événement à la mémoire SOUL

        Args:
            event_type: Type d'événement (ex: "skill_execution", "api_call", "error")
            description: Description human-readable
            metadata: Données additionnelles (dict)
        """
        status = self._load_status()

        if "custom_events" not in status:
            status["custom_events"] = []

        event = {
            "type": event_type,
            "description": description,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }

        status["custom_events"].append(event)

        # Limiter à 1000 événements max (keep last 1000)
        if len(status["custom_events"]) > 1000:
            status["custom_events"] = status["custom_events"][-1000:]

        self._save_status(status)

    def get_events(
        self,
        filter_type: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """
        Récupérer les événements de la mémoire SOUL

        Args:
            filter_type: Filtrer par type d'événement
            since: Timestamp minimum
            limit: Nombre maximum d'événements à retourner

        Returns:
            Liste d'événements
        """
        status = self._load_status()
        events = status.get("custom_events", [])

        # Filtrer par type
        if filter_type:
            events = [e for e in events if e.get("type") == filter_type]

        # Filtrer par date
        if since:
            events = [
                e for e in events
                if datetime.fromisoformat(e["timestamp"]) >= since
            ]

        # Limiter le nombre
        if limit:
            events = events[-limit:]  # Les plus récents

        return events

    def get_current_context(self) -> Dict[str, Any]:
        """
        Obtenir le contexte complet actuel

        Returns:
            Dict avec status, events récents, et contexte Git
        """
        status = self._load_status()

        # Événements des dernières 24h
        since_24h = datetime.now() - timedelta(hours=24)
        recent_events = self.get_events(since=since_24h)

        # Grouper les événements par type
        events_by_type = {}
        for event in recent_events:
            event_type = event.get("type", "unknown")
            if event_type not in events_by_type:
                events_by_type[event_type] = []
            events_by_type[event_type].append(event)

        return {
            "status": status.get("session_info", {}),
            "recent_events": recent_events,
            "events_by_type": events_by_type,
            "total_events_24h": len(recent_events),
            "timestamp": datetime.now().isoformat()
        }

    def get_session_summary(self) -> Dict[str, Any]:
        """
        Résumé de la session actuelle

        Returns:
            Dict avec résumé de la session
        """
        status = self._load_status()

        # Compter les événements par type
        events = status.get("custom_events", [])
        event_counts = {}
        for event in events:
            event_type = event.get("type", "unknown")
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

        return {
            "total_events": len(events),
            "event_counts": event_counts,
            "git_status": status.get("session_info", {}).get("git_clean", True),
            "last_event": events[-1] if events else None,
            "timestamp": datetime.now().isoformat()
        }

    def get_pattern_analysis(self, days: int = 7, threshold: int = 5) -> Dict[str, Any]:
        """
        Analyser les patterns d'utilisation pour NEXUS

        Args:
            days: Nombre de jours à analyser
            threshold: Seuil minimum pour considérer un pattern

        Returns:
            Dict avec patterns détectés
        """
        since = datetime.now() - timedelta(days=days)
        events = self.get_events(since=since)

        # Compter par type
        type_counts = {}
        type_contexts = {}

        for event in events:
            event_type = event.get("type", "unknown")
            type_counts[event_type] = type_counts.get(event_type, 0) + 1

            # Collecter les contextes
            if event_type not in type_contexts:
                type_contexts[event_type] = []

            type_contexts[event_type].append({
                "description": event.get("description"),
                "metadata": event.get("metadata", {}),
                "timestamp": event.get("timestamp")
            })

        # Filtrer les patterns qui dépassent le seuil
        patterns = {}
        for event_type, count in type_counts.items():
            if count >= threshold:
                patterns[event_type] = {
                    "count": count,
                    "frequency": count / days,  # Par jour
                    "contexts": type_contexts[event_type][:10],  # 10 exemples max
                    "suggested_skill": self._suggest_skill_name(event_type),
                    "priority": self._calculate_priority(count, days)
                }

        return {
            "analysis_period_days": days,
            "threshold": threshold,
            "patterns_detected": len(patterns),
            "patterns": patterns,
            "timestamp": datetime.now().isoformat()
        }

    def _suggest_skill_name(self, event_type: str) -> str:
        """Suggérer un nom de skill basé sur le type d'événement"""
        # Mapping simple type → skill name
        suggestions = {
            "api_call": "api-optimizer",
            "data_processing": "data-transformer",
            "file_operation": "file-handler",
            "database_query": "db-wizard",
            "test_execution": "test-guardian",
            "deployment": "deploy-sage",
            "error": "error-resolver"
        }

        return suggestions.get(event_type, f"{event_type.replace('_', '-')}-skill")

    def _calculate_priority(self, count: int, days: int) -> str:
        """Calculer la priorité basée sur la fréquence"""
        frequency = count / days

        if frequency >= 3:  # 3+ fois par jour
            return "critical"
        elif frequency >= 1:  # 1+ fois par jour
            return "high"
        elif frequency >= 0.5:  # 3-4 fois par semaine
            return "medium"
        else:
            return "low"

    def get_handoff_context(self) -> str:
        """
        Lire le contexte de handoff actuel

        Returns:
            Contenu du fichier .agent_handoff.md
        """
        if self.handoff_file.exists():
            with open(self.handoff_file, 'r') as f:
                return f.read()
        return ""

    def append_to_log(self, content: str):
        """
        Ajouter du contenu au log SOUL

        Args:
            content: Contenu à ajouter (markdown)
        """
        with open(self.log_file, 'a') as f:
            # Lock pour écriture
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write("\n" + content + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


# Fonctions globales pour faciliter l'utilisation

_soul_instance = None

def get_soul_instance(repo_path: str = ".") -> SOULMemory:
    """Obtenir l'instance SOUL singleton"""
    global _soul_instance
    if _soul_instance is None:
        _soul_instance = SOULMemory(repo_path)
    return _soul_instance


def add_soul_event(event_type: str, description: str, metadata: Optional[Dict] = None):
    """
    Ajouter un événement à la mémoire SOUL

    Exemple:
        add_soul_event(
            "api_call",
            "Called external API endpoint",
            {"endpoint": "/users", "method": "GET"}
        )
    """
    soul = get_soul_instance()
    soul.add_event(event_type, description, metadata)


def get_soul_memory(
    filter_type: Optional[str] = None,
    since: Optional[datetime] = None,
    limit: Optional[int] = None
) -> List[Dict]:
    """
    Récupérer les événements de la mémoire SOUL

    Exemple:
        # Tous les événements API des 7 derniers jours
        api_events = get_soul_memory(
            filter_type="api_call",
            since=datetime.now() - timedelta(days=7)
        )
    """
    soul = get_soul_instance()
    return soul.get_events(filter_type, since, limit)


def get_current_context() -> Dict[str, Any]:
    """
    Obtenir le contexte complet actuel

    Returns:
        Dict avec status, events récents, et contexte
    """
    soul = get_soul_instance()
    return soul.get_current_context()


def get_session_summary() -> Dict[str, Any]:
    """
    Résumé de la session actuelle

    Returns:
        Dict avec résumé de la session
    """
    soul = get_soul_instance()
    return soul.get_session_summary()


def get_pattern_analysis(days: int = 7, threshold: int = 5) -> Dict[str, Any]:
    """
    Analyser les patterns pour NEXUS

    Args:
        days: Nombre de jours à analyser
        threshold: Seuil minimum pour considérer un pattern

    Returns:
        Dict avec patterns détectés
    """
    soul = get_soul_instance()
    return soul.get_pattern_analysis(days, threshold)


def get_handoff_context() -> str:
    """Lire le contexte de handoff actuel"""
    soul = get_soul_instance()
    return soul.get_handoff_context()


def append_to_soul_log(content: str):
    """Ajouter du contenu au log SOUL"""
    soul = get_soul_instance()
    soul.append_to_log(content)


# Exemple d'utilisation
if __name__ == "__main__":
    # Test de l'API
    print("Testing SOUL API...")

    # Ajouter quelques événements
    add_soul_event(
        "api_call",
        "Test API call to external service",
        {"endpoint": "/test", "status": 200}
    )

    add_soul_event(
        "data_processing",
        "Processed CSV file",
        {"file": "data.csv", "rows": 1000}
    )

    # Récupérer les événements
    events = get_soul_memory(limit=10)
    print(f"Total events: {len(events)}")

    # Contexte actuel
    context = get_current_context()
    print(f"Events in last 24h: {context['total_events_24h']}")

    # Pattern analysis
    patterns = get_pattern_analysis(days=7, threshold=2)
    print(f"Patterns detected: {patterns['patterns_detected']}")

    for pattern_type, pattern_info in patterns['patterns'].items():
        print(f"  - {pattern_type}: {pattern_info['count']} occurrences")
        print(f"    Suggested skill: {pattern_info['suggested_skill']}")
        print(f"    Priority: {pattern_info['priority']}")

    print("\n✓ SOUL API test complete!")
