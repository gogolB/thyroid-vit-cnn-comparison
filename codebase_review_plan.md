# Codebase Improvement Review Plan

This document outlines the plan for reviewing the "Thyroid Classification Project" codebase and generating structured suggestions for improvements.

## Review Categories and Approach

The review and suggestions will be structured based on the following categories.

### 1. Initial Model Check (InceptionV3 Warning)
*   **Current Understanding:** The warning "Warning: InceptionV3 not found or could not be imported for registration in src.models.cnn" suggests an issue with how `InceptionV3` is being handled, potentially in its registration or instantiation via `timm`.
*   **Review Focus:** Examine `src/models/cnn/inception.py` and its interaction with `src/models/registry.py` and `timm`.
*   **Suggestion Goal:** Provide concrete steps to diagnose and resolve this warning, ensuring `InceptionV3` (and potentially other `timm` models) are correctly registered and usable.

### 2. Redundancy Reduction
*   **Current Understanding:** The refactoring aimed to reduce redundancy, particularly with the unified `ExperimentRunner` and `ModelBase`. However, opportunities might still exist, especially in LightningModules or utility functions.
*   **Review Focus:** Look for duplicated logic or patterns in `src/training/lightning_modules.py` (comparing `ThyroidCNNModule`, `ThyroidViTModule`, `ThyroidDistillationModule`), model definitions, and experiment setup.
*   **Suggestion Goal:** Identify specific areas where code can be further consolidated, perhaps through shared base classes for LightningModules, more comprehensive utility functions, or enhanced configuration-driven behavior.

### 3. Maintainability
*   **Current Understanding:** Key components like `ModelRegistry`, `ModelBase`, Pydantic schemas (`src/config/schemas.py`), and `ThyroidDataModule` (`src/data/datamodule.py`) are in place to improve maintainability.
*   **Review Focus:** Assess the effectiveness of current abstractions, the clarity and completeness of the configuration system (Pydantic schemas and Hydra config structure), the separation of concerns (e.g., in `scripts/experiment_runner.py`), and error handling mechanisms.
*   **Suggestion Goal:** Propose enhancements to existing abstractions, advocate for more configuration-driven components (e.g., making optimizer/scheduler choices in Lightning modules more dynamic), improve error reporting, and ensure clear boundaries between different parts of the codebase.

### 4. Usability (Developer Experience)
*   **Current Understanding:** The developer experience involves running experiments, adding new models, and general debugging.
*   **Review Focus:** Consider the ease of configuring and running experiments via `scripts/experiment_runner.py`, the process for integrating new models using `ModelBase` and `ModelRegistry`, and the clarity of logs and debugging information.
*   **Suggestion Goal:** Recommend ways to streamline experiment configuration, provide clearer guidelines or templates for adding new models, standardize logging for better traceability, and suggest potential helper scripts for common development tasks.

### 5. Readability
*   **Current Understanding:** The project uses Pylint, and type hints are present.
*   **Review Focus:** Assess naming conventions (variables, functions, classes), the adequacy of comments for complex logic, and overall code formatting in the examined files.
*   **Suggestion Goal:** Offer general recommendations for consistent naming, suggest areas where more explanatory comments could be beneficial, and reinforce the use of type hints and linters.

### 6. Future-Proofing & Extensibility

#### 6.1. CNN-ViT Hybrids
*   **Review Focus:** How would a hybrid model integrate with `ModelBase`, `ModelRegistry`, and the configuration system?
*   **Suggestion Goal:** Propose a potential structure for hybrid models (e.g., a `HybridBase` class, conventions for configuration, a new `model_type` in the registry).

#### 6.2. Multimodal Imaging
*   **Review Focus:** How can `src/data/datamodule.py` and `src/data/dataset.py` be adapted to handle multiple image inputs per sample? The `refactoring-guide.md` anticipates a `src/data/multimodal/dataset.py`.
*   **Suggestion Goal:** Outline necessary modifications to `DatasetConfig`, `CARSThyroidDataset.__getitem__` (e.g., returning a dictionary of images), `ThyroidDataModule`, and how models would consume these multiple inputs.

#### 6.3. Fusion Strategies (Early, Late, Intermediate)
*   **Review Focus:** Where would the logic for different fusion strategies reside? The `refactoring-guide.md` points towards a future `src/models/fusion/strategies.py`.
*   **Suggestion Goal:** Discuss architectural patterns for each fusion type, their implications for `ModelRegistry`, `ExperimentRunner`, and configuration schemas.

## Process
Each point will be analyzed by synthesizing information from the provided codebase files, the overall project structure, the `refactoring-guide.md`, and the user's request. This will lead to the formulation of specific, actionable suggestions.