"""Find conservative env-based URL hosts in YAML and dotenv config records."""

from pathlib import PurePosixPath

import yaml

from .url_eval import host_from_text, redact_userinfo


def _add(table, name, value, path, kind):
    if not isinstance(name, str) or not name:
        return
    value = None if value is None or isinstance(value, (dict, list)) else str(value)
    table.setdefault(name, []).append({"value": value, "path": path, "source": kind})


def _dotenv(text, path, table):
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, separator, value = line.partition("=")
        if not separator:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] in "\"'" and value[-1] == value[0]:
            value = value[1:-1]
        _add(table, name.strip(), value, path, "dotenv")


def _config_table(records, report):
    """Two passes: load complete YAML files, then resolve container references."""
    table, documents, configmaps = {}, [], {}
    for rec in records:
        path = "<unknown>"
        try:
            if not rec.is_config:
                continue
            report["config_files_total"] += 1
            path = rec.path
            name = PurePosixPath(path).name.lower()
            text = "\n".join(rec.source_lines)
            if name == ".env" or name.startswith(".env.") or name.endswith(".env"):
                _dotenv(text, path, table)
            elif PurePosixPath(path).suffix.lower() in (".yaml", ".yml"):
                # Materialize before committing any document: a later parse error
                # makes the entire file unknown, including earlier documents.
                try:
                    loaded = list(yaml.safe_load_all(text))
                except Exception as exc:
                    if path not in report["unparsed_files"]:
                        report["unparsed_files"].append(path)
                    report["unparsed_errors"][path] = f"{type(exc).__name__}: {exc}"
                    continue
                for doc in loaded:
                    if not isinstance(doc, dict):
                        continue
                    if doc.get("kind") == "Secret":
                        report["secret_documents"] += 1
                        continue
                    documents.append((path, doc))
                    if doc.get("kind") == "ConfigMap":
                        metadata = doc.get("metadata")
                        data = doc.get("data")
                        cm_name = metadata.get("name") if isinstance(metadata, dict) else None
                        if isinstance(cm_name, str) and isinstance(data, dict):
                            configmaps.setdefault(cm_name, []).append((path, data))
        except Exception as exc:
            report["file_failed"] += 1
            report["file_errors"].append(f"{path}: {type(exc).__name__}: {exc}")

    def configmap(ref):
        return configmaps.get(ref.get("name"), []) if isinstance(ref, dict) else []

    def container_env(container, path):
        if not isinstance(container, dict):
            return
        env = container.get("env", [])
        for item in env if isinstance(env, list) else []:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if "value" in item:
                _add(table, name, item["value"], path, "k8s_env")
                continue
            ref = item.get("valueFrom")
            if not isinstance(ref, dict):
                continue
            if "configMapKeyRef" in ref:
                key_ref = ref["configMapKeyRef"]
                key = key_ref.get("key") if isinstance(key_ref, dict) else None
                matches = [(cm_path, data[key]) for cm_path, data in configmap(key_ref) if key in data]
                if matches:
                    for cm_path, value in matches:
                        _add(table, name, value, cm_path, "k8s_configmap")
                else:
                    report["configmap_not_in_repo"] += 1
                    _add(table, name, None, path, "k8s_configmap")
            elif any(key in ref for key in ("secretKeyRef", "fieldRef", "resourceFieldRef")):
                report["secret_or_field_ref_skipped"] += 1
                _add(table, name, None, path, "k8s_env")
        env_from = container.get("envFrom", [])
        for item in env_from if isinstance(env_from, list) else []:
            if not isinstance(item, dict):
                continue
            if "secretRef" in item:
                report["secret_or_field_ref_skipped"] += 1
            elif "configMapRef" in item:
                matches = configmap(item["configMapRef"])
                if not matches:
                    report["configmap_not_in_repo"] += 1
                prefix = item.get("prefix", "")
                if not isinstance(prefix, str):
                    continue
                for cm_path, data in matches:
                    for name, value in data.items():
                        if isinstance(name, str):
                            _add(table, prefix + name, value, cm_path, "k8s_configmap")

    def walk(node, path, seen):
        if not isinstance(node, (dict, list)) or id(node) in seen:
            return
        seen.add(id(node))
        if isinstance(node, list):
            for child in node:
                walk(child, path, seen)
            return
        if node.get("kind") == "Secret":
            return
        for key in ("containers", "initContainers"):
            containers = node.get(key)
            if isinstance(containers, list):
                for container in containers:
                    container_env(container, path)
        for child in node.values():
            walk(child, path, seen)

    for path, doc in documents:
        try:
            services = doc.get("services")
            if isinstance(services, dict):
                for service in services.values():
                    if not isinstance(service, dict):
                        continue
                    environment = service.get("environment")
                    if isinstance(environment, dict):
                        for name, value in environment.items():
                            _add(table, name, value, path, "compose")
                    elif isinstance(environment, list):
                        for item in environment:
                            if isinstance(item, str) and "=" in item:
                                name, _, value = item.partition("=")
                                _add(table, name, value, path, "compose")
            # ConfigMap contents are values, not additional Pod specifications.
            if doc.get("kind") != "ConfigMap":
                walk(doc, path, set())
        except Exception as exc:
            report["file_failed"] += 1
            report["file_errors"].append(f"{path}: {type(exc).__name__}: {exc}")
    report["table_size"] = len(table)
    return table


def run_config_check(edges, records, literal_report, constprop_report, ctx=None):
    """Resolve only reported env references, without changing edge status."""
    report = {"edges_total": 0, "edges_considered": 0, "outcomes": {}, "samples": {},
              "resolved_edges": {}, "multi_value_edges": {}, "unparsed_files": [], "table_size": 0,
              "configmap_not_in_repo": 0, "secret_or_field_ref_skipped": 0, "secret_documents": 0,
              "config_files_total": 0, "unparsed_errors": {}, "value_not_url": {},
              "file_failed": 0, "file_errors": [], "edge_errors": {}, "reasons": []}
    try:
        report["edges_total"] = len(edges)
        table = _config_table(records, report)
        refs = dict((constprop_report or {}).get("env_refs", {}))
        refs.update((literal_report or {}).get("env_refs", {}))
        for index, ref in refs.items():
            report["edges_considered"] += 1
            edge, before = None, None
            url_reason = None
            sample = f"edge_index={index}"
            try:
                if type(index) is not int or not 0 <= index < len(edges):
                    raise IndexError(f"invalid edge index: {index}")
                edge = edges[index]
                sample = f"{edge.file}:{edge.line}"
                before = {key: getattr(edge, key) for key in ("config_matches", "host", "needs_review", "review_reason")}
                changes = {}
                if edge.status != "external_candidate" or edge.host is not None:
                    outcome = "skipped_not_candidate"
                elif ref["var"] is None:
                    outcome = "no_var_name"
                else:
                    entries = table.get(ref["var"], [])
                    valued = [entry for entry in entries if entry["value"] is not None]
                    values = list(dict.fromkeys(redact_userinfo(entry["value"]) for entry in valued))
                    selected = valued if len(values) == 1 else entries
                    paths = list(dict.fromkeys(entry["path"] for entry in selected))
                    if entries:
                        changes["config_matches"] = list(dict.fromkeys(edge.config_matches + paths))
                    if not entries:
                        outcome = "no_match"
                    elif not values:
                        outcome = "defined_value_unknown"
                    elif len(values) > 1:
                        outcome = "multiple_values"
                        report["multi_value_edges"][index] = {"values": values, "paths": paths}
                    else:
                        value = values[0]
                        cuts = [value.index(marker) for marker in ("$(", "${", "{{") if marker in value]
                        text = value[:min(cuts)] if cuts else value
                        host, reason = host_from_text(text, not cuts)
                        if host:
                            outcome = "config_host_set"
                            changes.update(host=host, needs_review=True,
                                           review_reason=edge.review_reason if edge.review_reason is not None else "config_value")
                            report["resolved_edges"][index] = {"host": host, "value": value, "paths": paths}
                        else:
                            outcome = "value_not_url"
                            url_reason = reason
                for key, value in changes.items():
                    setattr(edge, key, value)
                if url_reason is not None:
                    report["value_not_url"][url_reason] = report["value_not_url"].get(url_reason, 0) + 1
            except Exception as exc:
                if edge is not None and before is not None:
                    try:
                        vars(edge).update(before)
                    except (TypeError, AttributeError):
                        for key, value in before.items():
                            try:
                                setattr(edge, key, value)
                            except Exception:
                                pass
                report["resolved_edges"].pop(index, None)
                report["multi_value_edges"].pop(index, None)
                report["edge_errors"][index] = f"{type(exc).__name__}: {exc}"
                outcome = "edge_failed"
            report["outcomes"][outcome] = report["outcomes"].get(outcome, 0) + 1
            samples = report["samples"].setdefault(outcome, [])
            if sample not in samples and len(samples) < 5:
                samples.append(sample)
    except Exception as exc:
        report["reasons"].append(f"{type(exc).__name__}: {exc}")
    try:
        if ctx is not None:
            ctx.bump("config_check.edges_considered", report["edges_considered"])
            for outcome, count in report["outcomes"].items():
                ctx.bump(f"config_check.outcome.{outcome}", count)
    except Exception:
        pass
    return report
