"""
LangSmith utilities - Centralized tracing configuration
"""
import os
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from langsmith import Client

# Initialize LangSmith client
client = Client()

# Helper class to maintain compatibility with existing logger structure/syntax if needed,
# or to provide a clean interface for adding metadata to the current run.
class TracingContext:
    @staticmethod
    def update_current_observation(input=None, output=None, metadata=None, **kwargs):
        """
        Update the current run's metadata.
        This attempts to find the current run context managed by @traceable.
        """
        # In LangSmith, we typically rely on the @traceable decorator capturing inputs/outputs automatically.
        # For additional metadata usage or intermediate updates, we can use run_tree capabilities if accessible,
        # but the simple SDK usage via @traceable often suffices for inputs/outputs.
        
        # However, to support specific metadata logging similar to what was done with Langfuse:
        try:
            from langsmith.run_helpers import get_current_run_tree
            rt = get_current_run_tree()
            if rt:
                if metadata:
                    # Merge existing extra metadata
                    current_extra = rt.extra or {}
                    if "metadata" not in current_extra:
                        current_extra["metadata"] = {}
                    current_extra["metadata"].update(metadata)
                    rt.extra = current_extra
                    
                # We can also log intermediate outputs or inputs if really needed, 
                # though @traceable handles the function entry/exit params.
        except Exception:
            # If no active run tree, ignore
            pass

    @staticmethod
    def get_current_trace():
        """
        Get current trace ID if available.
        """
        try:
            from langsmith.run_helpers import get_current_run_tree
            rt = get_current_run_tree()
            if rt:
                # Return an object with an 'id' attribute to match previous usage
                return rt
        except Exception:
            return None
        return None

tracing_context = TracingContext()

__all__ = ['traceable', 'wrap_openai', 'tracing_context']
