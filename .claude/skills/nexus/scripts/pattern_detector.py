#!/usr/bin/env python3
"""NEXUS Pattern Detector - Simplified"""
import json

class PatternDetector:
    def __init__(self):
        self.patterns = {
            "api": ["api", "endpoint", "rest"],
            "data": ["csv", "json", "data"],
            "deploy": ["deploy", "docker"],
            "docs": ["readme", "documentation"],
            "test": ["test", "testing"],
            "workflow": ["workflow", "automation"]
        }
    
    def detect_patterns(self, text):
        scores = {}
        text_lower = text.lower()
        for pattern, keywords in self.patterns.items():
            score = sum(text_lower.count(k) for k in keywords)
            if score >= 3:
                scores[pattern] = score
        return scores

if __name__ == "__main__":
    detector = PatternDetector()
    print("NEXUS Pattern Detector ready")
