src.services.rag package
========================

Submodules
----------

src.services.rag.generator module
---------------------------------

.. automodule:: src.services.rag.generator
   :members:
   :show-inheritance:
   :undoc-members:

src.services.rag.models module
------------------------------

.. automodule:: src.services.rag.models
   :members:
   :show-inheritance:
   :undoc-members:

Module contents
---------------

.. automodule:: src.services.rag
   :members:
   :show-inheritance:
   :undoc-members:

Public package exports
----------------------

``src.services.rag`` exposes the production-ready surface for the RAG
pipeline, covering orchestration, retrieval, tracing, and utility helpers:

* ``RAGGenerator`` and ``LangGraphRAGPipeline`` for pipeline execution.
* ``RagTracingCallback`` to wire tracing spans into LangGraph workflows.
* ``VectorServiceRetriever`` with ``CompressionStats`` for vector access insights.
* ``RAGConfig``/``RAGRequest``/``RAGResult``/``RAGServiceMetrics``/``AnswerMetrics``
  and ``SourceAttribution`` for structured inputs and observability.
* ``build_default_rag_config`` and ``initialise_rag_generator`` for ergonomic setup.
