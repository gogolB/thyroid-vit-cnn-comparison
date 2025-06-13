#!/usr/bin/env python3
"""
Test runner script for Vision Transformer tests
Provides convenient commands for running different test suites
"""

import sys
import subprocess
import argparse
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

console = Console()

def run_command(cmd: list, description: str):
    """Run a command and display results"""
    console.print(f"\n[cyan]Running: {description}[/cyan]")
    console.print(f"[dim]Command: {' '.join(cmd)}[/dim]\n")
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode == 0:
        console.print(f"\n[green]✓ {description} passed![/green]")
    else:
        console.print(f"\n[red]✗ {description} failed![/red]")
        sys.exit(result.returncode)
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Run Vision Transformer tests")
    parser.add_argument(
        '--suite',
        choices=['all', 'unit', 'integration', 'vit', 'attention', 'quality'],
        default='all',
        help='Test suite to run'
    )
    parser.add_argument(
        '--coverage',
        action='store_true',
        help='Run with coverage report'
    )
    parser.add_argument(
        '--markers',
        type=str,
        help='Run tests with specific markers (e.g., "not slow")'
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--file',
        type=str,
        help='Run specific test file'
    )
    
    args = parser.parse_args()
    
    console.print(Panel.fit(
        "[bold cyan]Vision Transformer Test Runner[/bold cyan]\n"
        f"[dim]Suite: {args.suite}[/dim]",
        border_style="blue"
    ))
    
    # Base pytest command
    cmd = ["pytest"]
    
    # Add coverage if requested
    if args.coverage:
        cmd.extend(["--cov=src/models/vit", "--cov-report=html", "--cov-report=term"])
    
    # Add verbose if requested
    if args.verbose:
        cmd.append("-vv")
    
    # Add specific file if provided
    if args.file:
        cmd.append(args.file)
    else:
        # Add test selection based on suite
        if args.suite == 'unit':
            cmd.extend(["-m", "unit"])
        elif args.suite == 'integration':
            cmd.extend(["-m", "integration"])
        elif args.suite == 'vit':
            cmd.append("tests/test_vision_transformer_base.py")
        elif args.suite == 'attention':
            cmd.extend(["-m", "attention"])
        elif args.suite == 'quality':
            cmd.extend(["-m", "quality"])
        # 'all' runs everything
    
    # Add custom markers if provided
    if args.markers:
        cmd.extend(["-m", args.markers])
    
    # Run the tests
    run_command(cmd, f"{args.suite} tests")
    
    # Show coverage report location if generated
    if args.coverage:
        console.print("\n[cyan]Coverage report generated:[/cyan]")
        console.print("  HTML: htmlcov/index.html")
        console.print("  Run: python -m http.server 8000 --directory htmlcov")


if __name__ == "__main__":
    main()


# tests/test_vit_models.py
"""
Tests specifically for ViT model variants
This file will be populated when we implement vit_models.py
"""

import pytest
import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))


@pytest.mark.unit
class TestViTTiny:
    """Tests for ViT-Tiny model"""
    
    @pytest.mark.skip(reason="Implementation pending")
    def test_vit_tiny_creation(self):
        """Test ViT-Tiny model creation"""
        # TODO: Implement when vit_models.py is ready
        pass
    
    @pytest.mark.skip(reason="Implementation pending")
    def test_vit_tiny_parameters(self):
        """Test ViT-Tiny has expected parameter count"""
        # Expected: ~5.7M parameters
        pass


@pytest.mark.unit  
class TestViTSmall:
    """Tests for ViT-Small model"""
    
    @pytest.mark.skip(reason="Implementation pending")
    def test_vit_small_creation(self):
        """Test ViT-Small model creation"""
        pass
    
    @pytest.mark.skip(reason="Implementation pending")
    def test_vit_small_parameters(self):
        """Test ViT-Small has expected parameter count"""
        # Expected: ~22M parameters
        pass


@pytest.mark.integration
class TestViTIntegration:
    """Integration tests for ViT models"""
    
    @pytest.mark.skip(reason="Implementation pending")
    def test_pretrained_weight_loading(self):
        """Test loading pretrained weights"""
        pass
    
    @pytest.mark.skip(reason="Implementation pending")
    def test_unified_runner_integration(self):
        """Test integration with unified experiment runner"""
        pass


# Quick test to ensure imports work
def test_imports():
    """Test that all imports work correctly"""
    try:
        from src.models.vit.vision_transformer_base import VisionTransformerBase
        from src.models.vit.vision_transformer_base import PatchEmbed, Attention, Block
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")
