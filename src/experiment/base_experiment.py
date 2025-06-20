"""
Defines the base Experiment class.
"""
import abc
from typing import Any, Dict
import logging

from src.experiment.config import ExperimentConfig

logger = logging.getLogger(__name__)


class BaseExperiment(abc.ABC):
    """
    Abstract base class for all experiments.

    This class defines the common interface for setting up, running,
    and logging results of an experiment.
    """

    def __init__(self, config: ExperimentConfig):
        """
        Initializes the experiment with a given configuration.

        Args:
            config: The experiment configuration object.
        """
        self.config = config
        logger.info(f"Initializing experiment: {self.config.name}")

    @abc.abstractmethod
    def setup(self) -> None:
        """
        Sets up the experiment environment.
        This can include loading data, initializing models, etc.
        """
        pass

    @abc.abstractmethod
    def run(self) -> None:
        """
        Runs the core logic of the experiment.
        """
        pass

    @abc.abstractmethod
    def log_results(self) -> Dict[str, Any]:
        """
        Logs the results of the experiment.
        This could involve writing to files, databases, or experiment tracking tools.

        Returns:
            A dictionary containing key metrics and results.
        """
        pass

    def execute(self) -> Dict[str, Any]:
        """
        Executes the full experiment lifecycle: setup, run, and log_results.

        Returns:
            A dictionary containing key metrics and results from log_results.
        """
        logger.info(f"Starting setup for experiment: {self.config.name}")
        self.setup()
        logger.info(f"Setup complete. Running experiment: {self.config.name}")
        self.run()
        logger.info(f"Experiment run complete. Logging results for: {self.config.name}")
        results = self.log_results()
        logger.info(f"Results logged for experiment: {self.config.name}")
        return results