"""
Source modules for the Explainable AI News Integrity System

This package contains the core business logic modules:
- FakeNewsDetector: RoBERTa-based fake news classification
- LLMExplainer: Gemini-powered explanation generation
- ClaimExtractor: Simple single-prompt claim extraction
- ClaimifyExtractor: Three-stage pipeline claim extraction (higher quality)
- WikiRetriever: Wikipedia evidence retrieval via ChromaDB
- WikiRetrieverPG: Wikipedia evidence retrieval via PostgreSQL + pgvector
- PerplexityFactChecker: Web-search fact-checking via Perplexity
"""

from .classifier import FakeNewsDetector
from .explainer import LLMExplainer
from .extractor import ClaimExtractor
from .extractor_claimify import ClaimifyExtractor
from .retriever import WikiRetriever
from .retriever_pg import WikiRetrieverPG
from .perplexity_fact_checker import PerplexityFactChecker

__all__ = [
    # Classification
    'FakeNewsDetector',
    
    # Explanation
    'LLMExplainer',
    
    # Claim Extraction
    'ClaimExtractor',
    'ClaimifyExtractor',
    
    # Evidence Retrieval
    'WikiRetriever',
    'WikiRetrieverPG',
    
    # Fact Checking
    'PerplexityFactChecker',
]
