"""
Project Indexer for Orchestral Machine.

Generates a machine-readable project index (project_index.json) with:
- AST-based Python file analysis
- Incremental updates via mtime caching
- Graceful error handling
- Symbol search and dependency tracking

Usage:
    python -m src.tools.indexer
    or
    from src.tools.indexer import get_project_index, search_symbol, get_dependencies
"""

from __future__ import annotations

import ast
import fcntl
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


# --- Constants ---

INDEX_VERSION = "2.0"
EXCLUDE_DIRS = {"__pycache__", ".venv", ".git", "node_modules", "checkpoints", "snapshots", ".agent"}


# --- IndexCache ---


class IndexCache:
    """Cache for incremental updates based on file modification times."""

    def __init__(self, cache_path: Path):
        self.cache_path = cache_path
        self.file_mtimes: dict[str, float] = {}
        self._load()

    def should_reparse(self, filepath: Path) -> bool:
        """Check if file needs to be reparsed based on mtime."""
        try:
            current_mtime = filepath.stat().st_mtime
            cached_mtime = self.file_mtimes.get(str(filepath))
            return cached_mtime is None or cached_mtime != current_mtime
        except OSError:
            return True

    def update(self, filepath: Path) -> None:
        """Update cached mtime for a file."""
        try:
            self.file_mtimes[str(filepath)] = filepath.stat().st_mtime
        except OSError:
            pass

    def remove(self, filepath: Path) -> None:
        """Remove a file from cache."""
        self.file_mtimes.pop(str(filepath), None)

    def save(self) -> None:
        """Save cache to disk."""
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.file_mtimes, separators=(",", ":")))
        except OSError as e:
            logging.warning(f"Failed to save cache to {self.cache_path}: {e}")

    def _load(self) -> None:
        """Load cache from disk."""
        try:
            if self.cache_path.exists():
                self.file_mtimes = json.loads(self.cache_path.read_text())
        except (OSError, json.JSONDecodeError):
            self.file_mtimes = {}


# --- ErrorHandler ---


class ErrorHandler:
    """Centralized error handling with recovery strategies."""

    def __init__(self):
        self.errors: list[dict[str, Any]] = []

    def handle(self, filepath: Path, error_type: str, exception: Exception) -> dict[str, Any]:
        """Handle an error and return a fallback result."""
        error_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "filepath": str(filepath),
            "type": error_type,
            "message": str(exception),
        }
        self.errors.append(error_entry)

        return {
            "error": error_type,
            "message": str(exception),
            "filepath": str(filepath),
        }

    def has_errors(self) -> bool:
        """Check if any errors occurred."""
        return len(self.errors) > 0

    def get_errors(self) -> list[dict[str, Any]]:
        """Get all recorded errors."""
        return self.errors.copy()

    def get_summary(self) -> dict[str, Any]:
        """Get error summary."""
        by_type: dict[str, int] = {}
        for error in self.errors:
            error_type = error.get("type", "unknown")
            by_type[error_type] = by_type.get(error_type, 0) + 1

        return {
            "total_errors": len(self.errors),
            "by_type": by_type,
        }


# --- PythonFileAnalyzer ---


class PythonFileAnalyzer:
    """AST-based analyzer for Python files."""

    def __init__(self, error_handler: ErrorHandler):
        self.error_handler = error_handler

    def parse(self, filepath: Path) -> dict[str, Any]:
        """Parse a Python file and extract metadata."""
        try:
            source = self._read_file(filepath)
            tree = ast.parse(source)
            return self._extract_metadata(tree, filepath, source)
        except SyntaxError as e:
            return self.error_handler.handle(filepath, "syntax_error", e)
        except PermissionError as e:
            return self.error_handler.handle(filepath, "permission_denied", e)
        except UnicodeDecodeError as e:
            # Try fallback encoding
            try:
                source = self._read_file(filepath, encoding="latin-1")
                tree = ast.parse(source)
                return self._extract_metadata(tree, filepath, source)
            except Exception as fallback_e:
                return self.error_handler.handle(filepath, "encoding_error", fallback_e)
        except FileNotFoundError as e:
            return self.error_handler.handle(filepath, "file_not_found", e)
        except Exception as e:
            return self.error_handler.handle(filepath, "parse_error", e)

    def _read_file(self, filepath: Path, encoding: str = "utf-8") -> str:
        """Read file contents with specified encoding."""
        return filepath.read_text(encoding=encoding)

    def _extract_metadata(self, tree: ast.Module, filepath: Path, source: str) -> dict[str, Any]:
        """Extract metadata from AST."""
        try:
            metadata = {
                "summary": self._extract_docstring(tree),
                "classes": self._extract_classes(tree),
                "functions": self._extract_functions(tree),
                "imports": self._extract_imports(tree),
                "globals": self._extract_globals(tree),
            }
            # For __init__.py files, also extract __all__ exports
            if filepath.name == "__init__.py":
                metadata["exports"] = self._extract_exports(tree)
            return metadata
        except Exception as e:
            return self.error_handler.handle(filepath, "extraction_error", e)

    def _extract_exports(self, tree: ast.Module) -> dict[str, Any]:
        """Extract __all__ and re-exports from __init__.py files."""
        exports = {
            "__all__": [],
            "re_exports": [],
        }
        try:
            for node in ast.iter_child_nodes(tree):
                # Check for __all__ assignment
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == "__all__":
                            if isinstance(node.value, (ast.List, ast.Tuple)):
                                for elt in node.value.elts:
                                    if isinstance(elt, ast.Constant):
                                        exports["__all__"].append(elt.value)
                # Check for re-exports: from .module import name
                elif isinstance(node, ast.ImportFrom):
                    if node.module and (node.module.startswith(".") or node.module in ["src", "src.config", "src.nodes", "src.tools", "src.api"]):
                        for alias in node.names:
                            exports["re_exports"].append(alias.name)
        except Exception as e:
            logging.warning(f"Failed to extract exports: {e}")
        return exports

    def _extract_docstring(self, node: ast.Module | ast.ClassDef | ast.FunctionDef) -> str:
        """Extract module/class/function docstring."""
        docstring = ast.get_docstring(node)
        return docstring.split("\n")[0] if docstring else ""

    def _extract_classes(self, tree: ast.Module) -> dict[str, Any]:
        """Extract class definitions with methods, bases, decorators."""
        classes = {}
        try:
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.ClassDef):
                    classes[node.name] = {
                        "description": self._extract_docstring(node),
                        "bases": self._safe_unparse_list(node.bases),
                        "decorators": self._safe_unparse_list(node.decorator_list),
                        "methods": self._extract_methods(node),
                    }
        except Exception as e:
            logging.warning(f"Failed to extract classes: {e}")
        return classes

    def _extract_methods(self, class_node: ast.ClassDef) -> dict[str, Any]:
        """Extract methods from a class."""
        methods = {}
        try:
            for node in ast.iter_child_nodes(class_node):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods[node.name] = {
                        "signature": self._get_signature(node),
                        "is_async": isinstance(node, ast.AsyncFunctionDef),
                        "decorators": self._safe_unparse_list(node.decorator_list),
                    }
        except Exception:
            pass
        return methods

    def _extract_functions(self, tree: ast.Module) -> dict[str, Any]:
        """Extract top-level function definitions."""
        functions = {}
        try:
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    functions[node.name] = {
                        "signature": self._get_signature(node),
                        "is_async": isinstance(node, ast.AsyncFunctionDef),
                        "decorators": self._safe_unparse_list(node.decorator_list),
                    }
        except Exception:
            pass
        return functions

    def _get_signature(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
        """Get function signature as string."""
        try:
            args = []
            # Regular args
            for arg in node.args.args:
                arg_str = arg.arg
                if arg.annotation:
                    arg_str += f": {ast.unparse(arg.annotation)}"
                args.append(arg_str)

            # Varargs (*args)
            if node.args.vararg:
                arg_str = f"*{node.args.vararg.arg}"
                if node.args.vararg.annotation:
                    arg_str += f": {ast.unparse(node.args.vararg.annotation)}"
                args.append(arg_str)

            # Kwargs (**kwargs)
            if node.args.kwarg:
                arg_str = f"**{node.args.kwarg.arg}"
                if node.args.kwarg.annotation:
                    arg_str += f": {ast.unparse(node.args.kwarg.annotation)}"
                args.append(arg_str)

            # Return annotation
            return_annotation = ""
            if node.returns:
                return_annotation = f" -> {ast.unparse(node.returns)}"

            return f"({', '.join(args)}){return_annotation}"
        except Exception:
            return "(...)"

    def _extract_imports(self, tree: ast.Module) -> dict[str, Any]:
        """Extract import statements."""
        direct: list[str] = []
        from_imports: list[dict[str, Any]] = []

        try:
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        direct.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    names = [alias.name for alias in node.names]
                    from_imports.append({"module": module, "names": names})
        except Exception:
            pass

        return {"direct": direct, "from_imports": from_imports}

    def _extract_globals(self, tree: ast.Module) -> dict[str, Any]:
        """Extract module-level variable assignments."""
        globals_dict: dict[str, Any] = {}

        try:
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            type_hint = self._infer_type(node.value)
                            globals_dict[target.id] = {"type": type_hint}
                elif isinstance(node, ast.AnnAssign):
                    if isinstance(node.target, ast.Name):
                        type_hint = ast.unparse(node.annotation) if node.annotation else "Any"
                        globals_dict[node.target.id] = {"type": type_hint}
        except Exception:
            pass

        return globals_dict

    def _infer_type(self, node: ast.expr) -> str:
        """Infer type from an expression."""
        try:
            if isinstance(node, ast.Constant):
                return type(node.value).__name__
            elif isinstance(node, ast.List):
                return "list"
            elif isinstance(node, ast.Dict):
                return "dict"
            elif isinstance(node, ast.Set):
                return "set"
            elif isinstance(node, ast.Tuple):
                return "tuple"
            elif isinstance(node, ast.Name):
                return node.id
            else:
                return "Any"
        except Exception:
            return "Any"

    def _safe_unparse_list(self, nodes: list[ast.expr]) -> list[str]:
        """Safely unparse a list of AST nodes."""
        result = []
        for node in nodes:
            try:
                result.append(ast.unparse(node))
            except Exception:
                result.append("<unparseable>")
        return result


# --- IncrementalBuilder ---


class IncrementalBuilder:
    """Incremental index builder with change detection."""

    def __init__(self, root: Path, cache: IndexCache, error_handler: ErrorHandler):
        self.root = root
        self.cache = cache
        self.error_handler = error_handler
        self.analyzer = PythonFileAnalyzer(error_handler)

    def build(self, force_full: bool = False) -> dict[str, Any]:
        """Build index with incremental updates."""
        existing_index = self._load_existing_index() if not force_full else None

        if existing_index and not force_full:
            # Incremental update
            changed_files = self._get_changed_files()
            deleted_files = self._get_deleted_files(existing_index)

            if not changed_files and not deleted_files:
                # Nothing changed, return existing index with fresh timestamp
                existing_index["timestamp"] = datetime.now(timezone.utc).isoformat()
                return existing_index

            # Remove deleted files
            for filepath in deleted_files:
                existing_index["files"].pop(filepath, None)
                self.cache.remove(Path(filepath))

            # Update changed files
            for filepath in changed_files:
                relative_path = self._get_relative_path(filepath)
                file_data = self.analyzer.parse(filepath)
                existing_index["files"][relative_path] = file_data
                self.cache.update(filepath)

            # Rebuild derived data
            existing_index["timestamp"] = datetime.now(timezone.utc).isoformat()
            existing_index["stats"] = self._calculate_stats(existing_index)
            existing_index["symbols"] = self._build_symbol_index(existing_index)
            existing_index["dependency_graph"] = self._build_dependency_graph(existing_index)
            existing_index["errors"] = self.error_handler.get_errors()

            self.cache.save()
            return existing_index
        else:
            # Full build
            return self._full_build()

    def _full_build(self) -> dict[str, Any]:
        """Perform a full index build."""
        index: dict[str, Any] = {
            "version": INDEX_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "root_path": str(self.root),
            "stats": {},
            "files": {},
            "symbols": {},
            "dependency_graph": {},
            "errors": [],
        }

        # Collect all Python files
        for filepath in self._iter_python_files():
            relative_path = self._get_relative_path(filepath)
            file_data = self.analyzer.parse(filepath)
            index["files"][relative_path] = file_data
            self.cache.update(filepath)

        # Build derived data
        index["stats"] = self._calculate_stats(index)
        index["symbols"] = self._build_symbol_index(index)
        index["dependency_graph"] = self._build_dependency_graph(index)
        index["errors"] = self.error_handler.get_errors()

        self.cache.save()
        return index

    def _iter_python_files(self) -> list[Path]:
        """Iterate over all Python files in root, excluding certain directories."""
        files = []
        try:
            for filepath in self.root.rglob("*.py"):
                if self._should_include(filepath):
                    files.append(filepath)
        except Exception:
            pass
        return files

    def _should_include(self, filepath: Path) -> bool:
        """Check if a file should be included in the index."""
        try:
            # Check if any parent directory is in exclude list
            for part in filepath.parts:
                if part in EXCLUDE_DIRS:
                    return False
            return True
        except Exception:
            return False

    def _get_relative_path(self, filepath: Path) -> str:
        """Get path relative to root."""
        try:
            return str(filepath.relative_to(self.root.parent))
        except ValueError:
            return str(filepath)

    def _get_changed_files(self) -> list[Path]:
        """Get list of files that have changed since last index."""
        changed = []
        for filepath in self._iter_python_files():
            if self.cache.should_reparse(filepath):
                changed.append(filepath)
        return changed

    def _get_deleted_files(self, index: dict[str, Any]) -> list[str]:
        """Get list of files that were deleted since last index."""
        existing_files = {str(f) for f in self._iter_python_files()}
        indexed_files = set(index.get("files", {}).keys())

        # Convert relative paths for comparison
        deleted = []
        for filepath in indexed_files:
            full_path = self.root.parent / filepath
            if not full_path.exists():
                deleted.append(filepath)

        return deleted

    def _load_existing_index(self) -> Optional[dict[str, Any]]:
        """Load existing index from disk."""
        index_path = self.root.parent / "project_index.json"
        try:
            if index_path.exists():
                return json.loads(index_path.read_text())
        except (OSError, json.JSONDecodeError):
            pass
        return None

    def _calculate_stats(self, index: dict[str, Any]) -> dict[str, int]:
        """Calculate index statistics."""
        total_files = len(index.get("files", {}))
        total_classes = 0
        total_functions = 0

        for file_data in index.get("files", {}).values():
            if isinstance(file_data, dict) and "error" not in file_data:
                total_classes += len(file_data.get("classes", {}))
                total_functions += len(file_data.get("functions", {}))
                # Count methods as functions
                for class_data in file_data.get("classes", {}).values():
                    if isinstance(class_data, dict):
                        total_functions += len(class_data.get("methods", {}))

        return {
            "total_files": total_files,
            "total_classes": total_classes,
            "total_functions": total_functions,
            "total_errors": len(self.error_handler.get_errors()),
        }

    def _build_symbol_index(self, index: dict[str, Any]) -> dict[str, list[str]]:
        """Build symbol index for O(1) lookup."""
        symbols: dict[str, list[str]] = {}

        for filepath, file_data in index.get("files", {}).items():
            if isinstance(file_data, dict) and "error" not in file_data:
                # Index classes
                for class_name in file_data.get("classes", {}).keys():
                    key = f"class:{class_name}"
                    if key not in symbols:
                        symbols[key] = []
                    symbols[key].append(filepath)

                # Index functions
                for func_name in file_data.get("functions", {}).keys():
                    key = f"function:{func_name}"
                    if key not in symbols:
                        symbols[key] = []
                    symbols[key].append(filepath)

        return symbols

    def _build_dependency_graph(self, index: dict[str, Any]) -> dict[str, list[str]]:
        """Build dependency graph between files."""
        graph: dict[str, list[str]] = {}

        for filepath, file_data in index.get("files", {}).items():
            if isinstance(file_data, dict) and "error" not in file_data:
                deps = set()
                imports = file_data.get("imports", {})

                # Process direct imports
                for module in imports.get("direct", []):
                    dep_path = self._resolve_module_to_path(module)
                    if dep_path:
                        deps.add(dep_path)

                # Process from imports
                for from_import in imports.get("from_imports", []):
                    module = from_import.get("module", "")
                    dep_path = self._resolve_module_to_path(module)
                    if dep_path:
                        deps.add(dep_path)

                if deps:
                    graph[filepath] = sorted(list(deps))

        return graph

    def _resolve_module_to_path(self, module: str) -> Optional[str]:
        """Resolve a module name to a file path within the project."""
        # Convert module to potential file paths
        parts = module.split(".")

        # If module starts with "src.", strip it and search in self.root (src/)
        # This fixes the issue where "src.config" was incorrectly searched as "src/src/config.py"
        if parts[0] == "src":
            parts = parts[1:]

        if not parts:
            return None

        # Try as a direct file (search in self.root)
        potential_path = self.root.joinpath(*parts).with_suffix(".py")
        if potential_path.exists():
            return self._get_relative_path(potential_path)

        # Try as a package
        potential_path = self.root.joinpath(*parts, "__init__.py")
        if potential_path.exists():
            return self._get_relative_path(potential_path)

        return None


# --- File Locking ---


def _with_file_lock(filepath: Path, mode: str, content: str = "") -> None:
    """Write to file with locking to prevent race conditions."""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, mode) as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            if content:
                f.write(content)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


# --- Public API ---


def _get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def _get_src_root() -> Path:
    """Get the src directory."""
    return _get_project_root() / "src"


def _get_index_path() -> Path:
    """Get the index file path."""
    return _get_project_root() / "project_index.json"


def _get_cache_path() -> Path:
    """Get the cache file path."""
    return _get_project_root() / ".index_cache.json"


def _is_index_fresh(index_path: Path, ttl_minutes: int) -> bool:
    """Check if the index is fresh within TTL."""
    if not index_path.exists():
        return False

    try:
        index_data = json.loads(index_path.read_text())
        timestamp_str = index_data.get("timestamp", "")
        if not timestamp_str:
            return False

        timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        age = (datetime.now(timezone.utc) - timestamp).total_seconds() / 60

        return age < ttl_minutes
    except (OSError, json.JSONDecodeError, ValueError):
        return False


def get_project_index(regenerate: bool = False, ttl_minutes: int = 30) -> str:
    """
    Get the project index as a JSON string.

    Args:
        regenerate: Force full regeneration of the index
        ttl_minutes: Time-to-live for cached index in minutes

    Returns:
        Minified JSON string of the project index
    """
    index_path = _get_index_path()
    cache_path = _get_cache_path()

    # Check TTL
    if not regenerate and _is_index_fresh(index_path, ttl_minutes):
        try:
            return index_path.read_text()
        except OSError:
            pass

    # Build index
    error_handler = ErrorHandler()
    cache = IndexCache(cache_path)
    builder = IncrementalBuilder(_get_src_root(), cache, error_handler)
    index = builder.build(force_full=regenerate)

    # Save with file locking
    try:
        _with_file_lock(index_path, "w", json.dumps(index, separators=(",", ":")))
    except OSError:
        pass

    return json.dumps(index, separators=(",", ":"))


def search_symbol(query: str) -> list[dict[str, Any]]:
    """
    Search for a symbol in the project index.

    Args:
        query: Symbol name to search for (case-insensitive)

    Returns:
        List of matching symbols with their locations
    """
    index = json.loads(get_project_index())
    results = []
    query_lower = query.lower()

    for filepath, data in index.get("files", {}).items():
        if isinstance(data, dict) and "error" not in data:
            # Search in classes
            for class_name, class_data in data.get("classes", {}).items():
                if query_lower in class_name.lower():
                    results.append({
                        "type": "class",
                        "name": class_name,
                        "filepath": filepath,
                        "description": class_data.get("description", "") if isinstance(class_data, dict) else "",
                    })

            # Search in functions
            for func_name, func_data in data.get("functions", {}).items():
                if query_lower in func_name.lower():
                    results.append({
                        "type": "function",
                        "name": func_name,
                        "filepath": filepath,
                        "signature": func_data.get("signature", "") if isinstance(func_data, dict) else "",
                    })

    return results


def get_dependencies(filepath: str) -> dict[str, Any]:
    """
    Get dependencies for a specific file.

    Args:
        filepath: Path to the file (relative to project root)

    Returns:
        Dictionary with imports and dependents
    """
    index = json.loads(get_project_index())

    # Normalize filepath
    if not filepath.startswith("src/"):
        filepath = f"src/{filepath}"

    file_data = index.get("files", {}).get(filepath, {})

    # Find dependents (files that import this file)
    dependents = []
    for other_file, other_data in index.get("files", {}).items():
        if other_file == filepath:
            continue

        if isinstance(other_data, dict) and "error" not in other_data:
            imports = other_data.get("imports", {})

            # Check if this file imports the target
            for module in imports.get("direct", []):
                if filepath.replace("/", ".").replace(".py", "") in module:
                    dependents.append(other_file)

            for from_import in imports.get("from_imports", []):
                module = from_import.get("module", "")
                if filepath.replace("/", ".").replace(".py", "") in module:
                    dependents.append(other_file)

    return {
        "filepath": filepath,
        "imports": file_data.get("imports", {}) if isinstance(file_data, dict) else {},
        "dependents": list(set(dependents)),
    }


# --- CLI Entry Point ---


def main() -> None:
    """CLI entry point for the indexer."""
    print("Building project index...")
    index_json = get_project_index(regenerate=True)
    index = json.loads(index_json)

    print(f"Index version: {index.get('version')}")
    print(f"Timestamp: {index.get('timestamp')}")
    print(f"Root path: {index.get('root_path')}")
    print(f"Stats: {index.get('stats')}")
    print(f"Errors: {len(index.get('errors', []))}")

    index_path = _get_index_path()
    print(f"\nIndex saved to: {index_path}")


if __name__ == "__main__":
    main()
