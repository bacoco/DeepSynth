#!/usr/bin/env python3
"""
Script pour nettoyer les datasets obsolètes sur HuggingFace.

Usage:
    python cleanup_huggingface_datasets.py --list          # Liste les datasets
    python cleanup_huggingface_datasets.py --delete <name> # Supprime un dataset
    python cleanup_huggingface_datasets.py --delete-all    # Supprime tout (DANGER!)

Ce script vous aide à supprimer les vieux datasets de test qui ne servent plus.
"""

import os
import sys
import argparse
from typing import List

from huggingface_hub import HfApi, login, whoami, list_datasets
from dotenv import load_dotenv

# Load .env
load_dotenv()

class DatasetCleaner:
    def __init__(self):
        """Initialize with HuggingFace credentials"""
        self.token = os.getenv('HF_TOKEN')
        if not self.token:
            print("❌ HF_TOKEN not found in .env")
            sys.exit(1)

        login(token=self.token)
        self.username = whoami()['name']
        self.api = HfApi()

        print(f"✅ Logged in as: {self.username}")

    def list_my_datasets(self, pattern=None) -> List[str]:
        """List all datasets owned by the user"""
        print(f"\n📊 Listing datasets for user: {self.username}")

        try:
            datasets = list_datasets(author=self.username, use_auth_token=self.token)
            dataset_names = []

            for ds in datasets:
                name = ds.id
                if pattern and pattern not in name:
                    continue

                # Get size if possible
                try:
                    info = self.api.dataset_info(name, token=self.token)
                    # Approximate size from number of files
                    files = len(getattr(info, 'siblings', []))
                    size_info = f"({files} files)"
                except:
                    size_info = ""

                dataset_names.append(name)
                print(f"  📦 {name} {size_info}")

            print(f"\n✅ Total: {len(dataset_names)} datasets")
            return dataset_names

        except Exception as e:
            print(f"❌ Error listing datasets: {e}")
            return []

    def delete_dataset(self, dataset_name: str, confirm: bool = True):
        """Delete a single dataset"""
        # Add username prefix if not present
        if '/' not in dataset_name:
            dataset_name = f"{self.username}/{dataset_name}"

        print(f"\n🗑️  Preparing to delete: {dataset_name}")

        if confirm:
            print("⚠️  WARNING: This action CANNOT be undone!")
            response = input(f"Type the dataset name to confirm deletion: ")

            if response != dataset_name.split('/')[-1]:
                print("❌ Name mismatch - deletion aborted")
                return False

        try:
            self.api.delete_repo(repo_id=dataset_name, repo_type="dataset", token=self.token)
            print(f"✅ Successfully deleted: {dataset_name}")
            return True

        except Exception as e:
            print(f"❌ Error deleting {dataset_name}: {e}")
            return False

    def delete_multiple(self, patterns: List[str], confirm: bool = True):
        """Delete multiple datasets matching patterns"""
        all_datasets = self.list_my_datasets()

        to_delete = []
        for pattern in patterns:
            matching = [ds for ds in all_datasets if pattern in ds]
            to_delete.extend(matching)

        # Remove duplicates
        to_delete = list(set(to_delete))

        if not to_delete:
            print("❌ No datasets match the given patterns")
            return

        print(f"\n🗑️  Found {len(to_delete)} datasets to delete:")
        for ds in to_delete:
            print(f"  - {ds}")

        if confirm:
            print("\n⚠️  WARNING: This will DELETE ALL listed datasets!")
            response = input(f"Type 'DELETE ALL' to confirm: ")

            if response != 'DELETE ALL':
                print("❌ Deletion aborted")
                return

        # Delete each dataset
        success = 0
        failed = 0

        for ds in to_delete:
            if self.delete_dataset(ds, confirm=False):
                success += 1
            else:
                failed += 1

        print(f"\n📊 Results: {success} deleted, {failed} failed")

    def interactive_mode(self):
        """Interactive menu for dataset cleanup"""
        while True:
            print("\n" + "=" * 60)
            print("🧹 HUGGINGFACE DATASET CLEANUP")
            print("=" * 60)
            print("1. List all my datasets")
            print("2. List datasets matching pattern")
            print("3. Delete a single dataset")
            print("4. Delete multiple datasets (by pattern)")
            print("5. Exit")
            print()

            choice = input("Choose an option (1-5): ").strip()

            if choice == '1':
                self.list_my_datasets()

            elif choice == '2':
                pattern = input("Enter pattern to match: ").strip()
                self.list_my_datasets(pattern=pattern)

            elif choice == '3':
                dataset_name = input("Enter dataset name to delete: ").strip()
                self.delete_dataset(dataset_name)

            elif choice == '4':
                pattern = input("Enter pattern to match (e.g., 'test', 'old'): ").strip()
                self.delete_multiple([pattern])

            elif choice == '5':
                print("👋 Goodbye!")
                break

            else:
                print("❌ Invalid option")


def main():
    parser = argparse.ArgumentParser(description="Clean up HuggingFace datasets")
    parser.add_argument('--list', action='store_true', help='List all datasets')
    parser.add_argument('--pattern', type=str, help='Filter by pattern')
    parser.add_argument('--delete', type=str, help='Delete a specific dataset')
    parser.add_argument('--delete-pattern', type=str, help='Delete datasets matching pattern')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--yes', action='store_true', help='Skip confirmation prompts (DANGEROUS!)')

    args = parser.parse_args()

    cleaner = DatasetCleaner()

    # Interactive mode (default if no args)
    if len(sys.argv) == 1 or args.interactive:
        cleaner.interactive_mode()
        return

    # List datasets
    if args.list:
        cleaner.list_my_datasets(pattern=args.pattern)

    # Delete single dataset
    elif args.delete:
        cleaner.delete_dataset(args.delete, confirm=not args.yes)

    # Delete by pattern
    elif args.delete_pattern:
        cleaner.delete_multiple([args.delete_pattern], confirm=not args.yes)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
